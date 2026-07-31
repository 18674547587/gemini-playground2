// Author: PublicAffairs
// Project: https://github.com/PublicAffairs/openai-gemini
// MIT License : https://github.com/PublicAffairs/openai-gemini/blob/main/LICENSE
//
// === 增强版说明（在原始 OpenAI→Gemini 转换基础上补全原生能力）===
//  1. Function Calling 全链路：tools / tool_choice / role:"tool" 消息（functionResponse）
//  2. 兼容旧版 OpenAI functions 参数
//  3. 流式（SSE）下的 functionCall 增量输出（tool_calls delta）
//  4. 并行函数调用（单个响应多个 tool_calls）
//  5. thinking（思考）配置透传（Gemini 3 原生能力，可选）
//  6. 默认模型更新为 gemini-3-flash（2.x / 1.5 系列已退役返回 404）
//  7. 不再修改调用方传入的 messages 对象（消除副作用）

import { Buffer } from "node:buffer";

export default {
  async fetch (request) {
    if (request.method === "OPTIONS") {
      return handleOPTIONS();
    }
    const errHandler = (err) => {
      console.error(err);
      return new Response(err.message, fixCors({ status: err.status ?? 500 }));
    };
    try {
      const auth = request.headers.get("Authorization");
      const apiKey = auth?.split(" ")[1];
      const assert = (success) => {
        if (!success) {
          throw new HttpError("The specified HTTP method is not allowed for the requested resource", 400);
        }
      };
      const { pathname } = new URL(request.url);
      switch (true) {
        case pathname.endsWith("/chat/completions"):
          assert(request.method === "POST");
          return handleCompletions(await request.json(), apiKey)
            .catch(errHandler);
        case pathname.endsWith("/embeddings"):
          assert(request.method === "POST");
          return handleEmbeddings(await request.json(), apiKey)
            .catch(errHandler);
        case pathname.endsWith("/models"):
          assert(request.method === "GET");
          return handleModels(apiKey)
            .catch(errHandler);
        default:
          throw new HttpError("404 Not Found", 404);
      }
    } catch (err) {
      return errHandler(err);
    }
  }
};

class HttpError extends Error {
  constructor(message, status) {
    super(message);
    this.name = this.constructor.name;
    this.status = status;
  }
}

const fixCors = ({ headers, status, statusText }) => {
  headers = new Headers(headers);
  headers.set("Access-Control-Allow-Origin", "*");
  return { headers, status, statusText };
};

const handleOPTIONS = async () => {
  return new Response(null, {
    headers: {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "*",
      "Access-Control-Allow-Headers": "*",
    }
  });
};

const BASE_URL = "https://generativelanguage.googleapis.com";
const API_VERSION = "v1beta";

// https://github.com/google-gemini/generative-ai-js/blob/cf223ff4a1ee5a2d944c53cddb8976136382bee6/src/requests/request.ts#L71
const API_CLIENT = "genai-js/0.21.0"; // npm view @google/generative-ai version
const makeHeaders = (apiKey, more) => ({
  "x-goog-api-client": API_CLIENT,
  ...(apiKey && { "x-goog-api-key": apiKey }),
  ...more
});

async function handleModels (apiKey) {
  const response = await fetch(`${BASE_URL}/${API_VERSION}/models`, {
    headers: makeHeaders(apiKey),
  });
  let { body } = response;
  if (response.ok) {
    const { models } = JSON.parse(await response.text());
    body = JSON.stringify({
      object: "list",
      data: models.map(({ name }) => ({
        id: name.replace("models/", ""),
        object: "model",
        created: 0,
        owned_by: "",
      })),
    }, null, "  ");
  }
  return new Response(body, fixCors(response));
}

const DEFAULT_EMBEDDINGS_MODEL = "text-embedding-004";
async function handleEmbeddings (req, apiKey) {
  if (typeof req.model !== "string") {
    throw new HttpError("model is not specified", 400);
  }
  if (!Array.isArray(req.input)) {
    req.input = [ req.input ];
  }
  let model;
  if (req.model.startsWith("models/")) {
    model = req.model;
  } else {
    req.model = DEFAULT_EMBEDDINGS_MODEL;
    model = "models/" + req.model;
  }
  const response = await fetch(`${BASE_URL}/${API_VERSION}/${model}:batchEmbedContents`, {
    method: "POST",
    headers: makeHeaders(apiKey, { "Content-Type": "application/json" }),
    body: JSON.stringify({
      "requests": req.input.map(text => ({
        model,
        content: { parts: { text } },
        outputDimensionality: req.dimensions,
      }))
    })
  });
  let { body } = response;
  if (response.ok) {
    const { embeddings } = JSON.parse(await response.text());
    body = JSON.stringify({
      object: "list",
      data: embeddings.map(({ values }, index) => ({
        object: "embedding",
        index,
        embedding: values,
      })),
      model: req.model,
    }, null, "  ");
  }
  return new Response(body, fixCors(response));
}

// 默认模型更新：2.x / 1.5 系列已退役（404），3.x 免费可用
const DEFAULT_MODEL = "gemini-3-flash";
async function handleCompletions (req, apiKey) {
  let model = DEFAULT_MODEL;
  switch(true) {
    case typeof req.model !== "string":
      break;
    case req.model.startsWith("models/"):
      model = req.model.substring(7);
      break;
    case req.model.startsWith("gemini-"):
    case req.model.startsWith("learnlm-"):
      model = req.model;
  }
  const TASK = req.stream ? "streamGenerateContent" : "generateContent";
  let url = `${BASE_URL}/${API_VERSION}/models/${model}:${TASK}`;
  if (req.stream) { url += "?alt=sse"; }
  const response = await fetch(url, {
    method: "POST",
    headers: makeHeaders(apiKey, { "Content-Type": "application/json" }),
    body: JSON.stringify(await transformRequest(req)),
  });

  let body = response.body;
  if (response.ok) {
    let id = generateChatcmplId(); //"chatcmpl-8pMMaqXMK68B3nyDBrapTDrhkHBQK";
    if (req.stream) {
      body = response.body
        .pipeThrough(new TextDecoderStream())
        .pipeThrough(new TransformStream({
          transform: parseStream,
          flush: parseStreamFlush,
          buffer: "",
        }))
        .pipeThrough(new TransformStream({
          transform: toOpenAiStream,
          flush: toOpenAiStreamFlush,
          streamIncludeUsage: req.stream_options?.include_usage,
          model, id, last: [], toolCallIndex: 0,
        }))
        .pipeThrough(new TextEncoderStream());
    } else {
      body = await response.text();
      body = processCompletionsResponse(JSON.parse(body), model, id);
    }
  }
  return new Response(body, fixCors(response));
}

const harmCategory = [
  "HARM_CATEGORY_HATE_SPEECH",
  "HARM_CATEGORY_SEXUALLY_EXPLICIT",
  "HARM_CATEGORY_DANGEROUS_CONTENT",
  "HARM_CATEGORY_HARASSMENT",
  "HARM_CATEGORY_CIVIC_INTEGRITY",
];
const safetySettings = harmCategory.map(category => ({
  category,
  threshold: "BLOCK_NONE",
}));
const fieldsMap = {
  stop: "stopSequences",
  n: "candidateCount", // not for streaming
  max_tokens: "maxOutputTokens",
  max_completion_tokens: "maxOutputTokens",
  temperature: "temperature",
  top_p: "topP",
  top_k: "topK", // non-standard
  frequency_penalty: "frequencyPenalty",
  presence_penalty: "presencePenalty",
};
const transformConfig = (req) => {
  let cfg = {};
  //if (typeof req.stop === "string") { req.stop = [req.stop]; } // no need
  for (let key in req) {
    const matchedKey = fieldsMap[key];
    if (matchedKey) {
      cfg[matchedKey] = req[key];
    }
  }
  if (req.response_format) {
    switch(req.response_format.type) {
      case "json_schema":
        cfg.responseSchema = req.response_format.json_schema?.schema;
        if (cfg.responseSchema && "enum" in cfg.responseSchema) {
          cfg.responseMimeType = "text/x.enum";
          break;
        }
        // eslint-disable-next-line no-fallthrough
      case "json_object":
        cfg.responseMimeType = "application/json";
        break;
      case "text":
        cfg.responseMimeType = "text/plain";
        break;
      default:
        throw new HttpError("Unsupported response_format.type", 400);
    }
  }
  // 可选：thinking 配置透传（Gemini 3 原生能力）
  // 请求体里可传 "thinking": {"thinkingBudget": 0} / {"thinkingLevel": "LOW"}
  // 或简化写法 "thinking": 0（数字 = thinkingBudget，0 即关闭思考）
  if (req.thinking !== undefined) {
    if (typeof req.thinking === "number") {
      cfg.thinkingConfig = { thinkingBudget: req.thinking };
    } else if (typeof req.thinking === "object" && req.thinking !== null) {
      cfg.thinkingConfig = req.thinking;
    }
  }
  return cfg;
};

const parseImg = async (url) => {
  let mimeType, data;
  if (url.startsWith("http://") || url.startsWith("https://")) {
    try {
      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`${response.status} ${response.statusText} (${url})`);
      }
      mimeType = response.headers.get("content-type");
      data = Buffer.from(await response.arrayBuffer()).toString("base64");
    } catch (err) {
      throw new Error("Error fetching image: " + err.toString());
    }
  } else {
    const match = url.match(/^data:(?<mimeType>.*?)(;base64)?,(?<data>.*)$/);
    if (!match) {
      throw new Error("Invalid image data: " + url);
    }
    ({ mimeType, data } = match.groups);
  }
  return {
    inlineData: {
      mimeType,
      data,
    },
  };
};

/**
 * 单条消息 → Gemini parts
 * 支持：
 *  - content 为字符串（system/user/assistant 文本）
 *  - content 为数组（多模态：text / image_url / input_audio）
 *  - assistant 消息携带 tool_calls → functionCall parts
 *  - content 为 null（带 tool_calls 的 assistant 消息）
 */
const transformMsg = async ({ role, content, tool_calls }) => {
  const parts = [];

  // assistant 消息里的 tool_calls → Gemini functionCall parts
  if (Array.isArray(tool_calls)) {
    for (const tc of tool_calls) {
      if (tc.type !== "function" || !tc.function?.name) { continue; }
      let args = {};
      if (typeof tc.function.arguments === "string") {
        try { args = JSON.parse(tc.function.arguments); } catch { args = {}; }
      } else if (tc.function.arguments && typeof tc.function.arguments === "object") {
        args = tc.function.arguments;
      }
      parts.push({ functionCall: { name: tc.function.name, args } });
    }
  }

  if (!Array.isArray(content)) {
    // system, user: string; assistant: string 或 null（Required unless tool_calls is specified.）
    if (content != null && content !== "") {
      parts.push({ text: content });
    }
    return { role, parts };
  }

  // user: 多模态内容数组
  for (const item of content) {
    switch (item.type) {
      case "text":
        parts.push({ text: item.text });
        break;
      case "image_url":
        parts.push(await parseImg(item.image_url.url));
        break;
      case "input_audio":
        parts.push({
          inlineData: {
            mimeType: "audio/" + item.input_audio.format,
            data: item.input_audio.data,
          }
        });
        break;
      default:
        throw new TypeError(`Unknown "content" item type: "${item.type}"`);
    }
  }
  if (content.every(item => item.type === "image_url")) {
    parts.push({ text: "" }); // to avoid "Unable to submit request because it must have a text parameter"
  }
  return { role, parts };
};

/**
 * messages 数组 → Gemini contents + system_instruction
 * 支持完整工具调用链路：
 *  - system → system_instruction
 *  - assistant（含 tool_calls）→ model + functionCall parts
 *  - tool（工具结果）→ user + functionResponse parts
 *  - user → user
 * 不再修改传入的 messages 对象（无副作用）。
 */
const transformMessages = async (messages) => {
  if (!messages) { return; }
  const contents = [];
  let system_instruction;
  // 记录 OpenAI tool_call_id → 函数名，用于把 tool 消息还原为 functionResponse
  const toolCallMap = new Map();

  for (const item of messages) {
    const { role } = item;
    if (role === "system") {
      const { parts } = await transformMsg(item);
      system_instruction = { parts };
    } else if (role === "assistant") {
      if (Array.isArray(item.tool_calls)) {
        for (const tc of item.tool_calls) {
          if (tc.id && tc.function?.name) {
            toolCallMap.set(tc.id, tc.function.name);
          }
        }
      }
      contents.push(await transformMsg({ ...item, role: "model" }));
    } else if (role === "tool") {
      // OpenAI: { role:"tool", tool_call_id, content } → Gemini functionResponse（user 角色）
      const name = item.name || toolCallMap.get(item.tool_call_id);
      let response = item.content;
      if (typeof response === "string") {
        try { response = JSON.parse(response); } catch { /* 保留字符串 */ }
      }
      contents.push({
        role: "user",
        parts: [{ functionResponse: { name, response } }],
      });
    } else {
      // user（含 role 缺失/未知的情况按 user 处理）
      contents.push(await transformMsg({ ...item, role: "user" }));
    }
  }

  if (system_instruction && contents.length === 0) {
    contents.push({ role: "model", parts: [{ text: " " }] });
  }
  //console.info(JSON.stringify(contents, 2));
  return { system_instruction, contents };
};

// === Function Calling 转换 ===

// OpenAI tool → Gemini functionDeclaration
const transformTool = (tool) => {
  if (tool.type !== "function" || !tool.function?.name) {
    throw new HttpError(`Unsupported tool: ${JSON.stringify(tool).slice(0, 200)}`, 400);
  }
  const { name, description, parameters } = tool.function;
  return {
    name,
    description: description ?? "",
    parameters: parameters ?? { type: "object", properties: {} },
  };
};

// OpenAI tools 数组 → Gemini tools（functionDeclarations）
const transformTools = (tools) => {
  if (!Array.isArray(tools) || tools.length === 0) { return; }
  return [{ functionDeclarations: tools.map(transformTool) }];
};

// OpenAI tool_choice → Gemini toolConfig
// "auto"（默认）→ 不传（Gemini 默认 AUTO）
// "none" → NONE | "required" → ANY
// {type:"function", function:{name}} → ANY + allowedFunctionNames
const transformToolChoice = (toolChoice) => {
  if (!toolChoice || toolChoice === "auto") { return; }
  if (typeof toolChoice === "string") {
    switch (toolChoice) {
      case "none": return { functionCallingConfig: { mode: "NONE" } };
      case "required": return { functionCallingConfig: { mode: "ANY" } };
      default: throw new HttpError(`Unsupported tool_choice: ${toolChoice}`, 400);
    }
  }
  if (toolChoice.type === "function") {
    const name = toolChoice.function?.name;
    if (!name) { throw new HttpError("tool_choice.function.name is required", 400); }
    return { functionCallingConfig: { mode: "ANY", allowedFunctionNames: [name] } };
  }
  throw new HttpError("Unsupported tool_choice", 400);
};

const transformRequest = async (req) => {
  // 兼容旧版 OpenAI functions 参数（无 tools 时）
  const tools = req.tools ??
    (Array.isArray(req.functions)
      ? req.functions.map(f => ({ type: "function", function: f }))
      : undefined);
  const geminiTools = transformTools(tools);
  const toolConfig = transformToolChoice(req.tool_choice);
  return {
    ...(await transformMessages(req.messages)),
    safetySettings,
    ...(geminiTools && { tools: geminiTools }),
    ...(toolConfig && { toolConfig }),
    generationConfig: transformConfig(req),
  };
};

const generateChatcmplId = () => {
  const characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
  const randomChar = () => characters[Math.floor(Math.random() * characters.length)];
  return "chatcmpl-" + Array.from({ length: 29 }, randomChar).join("");
};

const generateToolCallId = () => {
  const characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
  const randomChar = () => characters[Math.floor(Math.random() * characters.length)];
  return "call_" + Array.from({ length: 24 }, randomChar).join("");
};

const reasonsMap = { //https://ai.google.dev/api/rest/v1/GenerateContentResponse#finishreason
  //"FINISH_REASON_UNSPECIFIED": // Default value. This value is unused.
  "STOP": "stop",
  "MAX_TOKENS": "length",
  "SAFETY": "content_filter",
  "RECITATION": "content_filter",
  //"OTHER": "OTHER",
  // :"function_call",
};
const SEP = "\n\n|>";

/**
 * Gemini candidate → OpenAI message/delta
 * 支持：
 *  - 纯文本 → message.content
 *  - functionCall → message.tool_calls（finish_reason 改为 "tool_calls"）
 *  - 文本 + functionCall 并存 → content 与 tool_calls 同时输出
 *  - 无文本内容 → content 为 null（符合 OpenAI 规范）
 */
const transformCandidates = (key, cand) => {
  const parts = cand.content?.parts || [];
  const textParts = parts.filter(p => p.text);
  const callParts = parts.filter(p => p.functionCall);

  const message = {
    role: "assistant",
    content: textParts.length > 0 ? textParts.map(p => p.text).join(SEP) : null,
  };

  let finish_reason = reasonsMap[cand.finishReason] || cand.finishReason;
  if (callParts.length > 0) {
    message.tool_calls = callParts.map(p => ({
      id: generateToolCallId(),
      type: "function",
      function: {
        name: p.functionCall.name,
        arguments: JSON.stringify(p.functionCall.args ?? {}),
      },
    }));
    finish_reason = "tool_calls";
  }

  return {
    index: cand.index || 0, // 0-index is absent in new -002 models response
    [key]: message,
    logprobs: null,
    finish_reason,
  };
};

const transformCandidatesMessage = transformCandidates.bind(null, "message");

const transformUsage = (data) => ({
  completion_tokens: data.candidatesTokenCount,
  prompt_tokens: data.promptTokenCount,
  total_tokens: data.totalTokenCount
});

const processCompletionsResponse = (data, model, id) => {
  return JSON.stringify({
    id,
    choices: data.candidates.map(transformCandidatesMessage),
    created: Math.floor(Date.now()/1000),
    model,
    //system_fingerprint: "fp_69829325d0",
    object: "chat.completion",
    usage: transformUsage(data.usageMetadata),
  });
};

const responseLineRE = /^data: (.*)(?:\n\n|\r\r|\r\n\r\n)/;
async function parseStream (chunk, controller) {
  chunk = await chunk;
  if (!chunk) { return; }
  this.buffer += chunk;
  do {
    const match = this.buffer.match(responseLineRE);
    if (!match) { break; }
    controller.enqueue(match[1]);
    this.buffer = this.buffer.substring(match[0].length);
  } while (true); // eslint-disable-line no-constant-condition
}
async function parseStreamFlush (controller) {
  if (this.buffer) {
    console.error("Invalid data:", this.buffer);
    controller.enqueue(this.buffer);
  }
}

/**
 * Gemini 流式 chunk → OpenAI SSE delta
 * 支持流式 functionCall：tool_calls delta（index 跨 chunk 递增，符合 OpenAI 规范）
 * 收尾 chunk（stop）：finish_reason 按是否发生函数调用输出 "tool_calls" / "stop"
 */
function transformResponseStream (data, stop, first) {
  const cand = data.candidates[0];
  const parts = cand.content?.parts || [];
  const textParts = parts.filter(p => p.text);
  const callParts = parts.filter(p => p.functionCall);

  const delta = {};
  if (first) {
    delta.role = "assistant";
    delta.content = "";
  }
  if (textParts.length > 0) {
    delta.content = (delta.content ?? "") + textParts.map(p => p.text).join("");
  }
  if (callParts.length > 0) {
    delta.tool_calls = callParts.map(p => ({
      index: this.toolCallIndex++,
      id: generateToolCallId(),
      type: "function",
      function: {
        name: p.functionCall.name,
        arguments: JSON.stringify(p.functionCall.args ?? {}),
      },
    }));
  }

  const item = {
    index: cand.index || 0,
    delta,
    finish_reason: null,
  };

  if (stop) {
    const hasCall = callParts.length > 0;
    item.delta = {};
    item.finish_reason = hasCall ? "tool_calls" : "stop";
  }

  const output = {
    id: this.id,
    choices: [item],
    created: Math.floor(Date.now()/1000),
    model: this.model,
    //system_fingerprint: "fp_69829325d0",
    object: "chat.completion.chunk",
  };
  if (data.usageMetadata && this.streamIncludeUsage) {
    output.usage = stop ? transformUsage(data.usageMetadata) : null;
  }
  return "data: " + JSON.stringify(output) + delimiter;
}
const delimiter = "\n\n";
async function toOpenAiStream (chunk, controller) {
  const transform = transformResponseStream.bind(this);
  const line = await chunk;
  if (!line) { return; }
  let data;
  try {
    data = JSON.parse(line);
  } catch (err) {
    console.error(line);
    console.error(err);
    const length = this.last.length || 1; // at least 1 error msg
    const candidates = Array.from({ length }, (_, index) => ({
      finishReason: "error",
      content: { parts: [{ text: err }] },
      index,
    }));
    data = { candidates };
  }
  const cand = data.candidates[0];
  console.assert(data.candidates.length === 1, "Unexpected candidates count: %d", data.candidates.length);
  cand.index = cand.index || 0; // absent in new -002 models response
  if (!this.last[cand.index]) {
    controller.enqueue(transform(data, false, "first"));
  }
  this.last[cand.index] = data;
  if (cand.content) { // prevent empty data (e.g. when MAX_TOKENS)
    controller.enqueue(transform(data));
  }
}
async function toOpenAiStreamFlush (controller) {
  const transform = transformResponseStream.bind(this);
  if (this.last.length > 0) {
    for (const data of this.last) {
      controller.enqueue(transform(data, "stop"));
    }
    controller.enqueue("data: [DONE]" + delimiter);
  }
}

// 导出内部转换函数（仅用于测试与二次开发，不影响 default export）
export {
  transformTools,
  transformToolChoice,
  transformMessages,
  transformCandidates,
  transformConfig,
  transformRequest,
  generateToolCallId,
};
