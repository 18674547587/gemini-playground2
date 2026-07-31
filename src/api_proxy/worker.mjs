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

// 代理版本标识：用于确认部署是否运行最新代码
const PROXY_VERSION = "v4.3.0";

const fixCors = ({ headers, status, statusText }) => {
  headers = new Headers(headers);
  headers.set("Access-Control-Allow-Origin", "*");
  headers.set("X-Proxy-Version", PROXY_VERSION);
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
      const fc = { name: tc.function.name, args };
      // 透传 thought_signature：优先取自定义字段，其次从 tool_call id 解码
      // （应对客户端丢弃自定义字段但保留 id 的情况）
      const sig = getThoughtSignature(tc);
      if (sig) {
        fc.thought_signature = sig;
      }
      parts.push({ functionCall: fc });
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
            toolCallMap.set(tc.id, {
              name: tc.function.name,
              thought_signature: getThoughtSignature(tc),
            });
          }
        }
      }
      contents.push(await transformMsg({ ...item, role: "model" }));
    } else if (role === "tool") {
      // OpenAI: { role:"tool", tool_call_id, content } → Gemini functionResponse（user 角色）
      const meta = toolCallMap.get(item.tool_call_id) || {};
      const name = item.name || meta.name;
      let response = item.content;
      if (typeof response === "string") {
        try { response = JSON.parse(response); } catch { /* 保留字符串 */ }
      }
      const fcResp = { name, response };
      // 透传 thought_signature（Gemini 思考模型必需）
      if (item.thought_signature || meta.thought_signature) {
        fcResp.thought_signature = item.thought_signature || meta.thought_signature;
      }
      contents.push({
        role: "user",
        parts: [{ functionResponse: fcResp }],
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

// Gemini functionDeclaration.parameters 不支持的 JSON Schema 关键字
// （Gemini 仅支持子集：type/format/description/nullable/enum/items/properties/
//   required/minItems/maxItems/minProperties/maxProperties/minLength/maxLength/minimum/maximum）
const SCHEMA_UNSUPPORTED = new Set([
  "$schema", "$id", "$ref", "definitions", "examples", "example",
  "oneOf", "anyOf", "allOf", "not", "if", "then", "else",
  "patternProperties", "additionalItems", "contains", "propertyNames",
  "default", "title", "deprecated", "readOnly", "writeOnly",
  "contentMediaType", "contentEncoding",
]);

// 递归清洗 JSON Schema：
//  - const → enum（语义等价，Gemini 支持 enum）
//  - 删除 Gemini 不支持的字段
//  - 递归处理嵌套的 properties / items
const cleanSchema = (schema) => {
  if (Array.isArray(schema)) {
    return schema.map(cleanSchema);
  }
  if (schema && typeof schema === "object") {
    const out = {};
    for (const [key, value] of Object.entries(schema)) {
      if (key === "const") {
        // Gemini 不支持 const，转为等价的 enum
        out.enum = [value];
        continue;
      }
      if (SCHEMA_UNSUPPORTED.has(key)) {
        continue; // 直接删除不支持的字段
      }
      if (key === "properties" && value && typeof value === "object") {
        // 关键：properties 内每个属性自身的 schema 也要递归清洗
        const cleaned = {};
        for (const [propName, propSchema] of Object.entries(value)) {
          cleaned[propName] = cleanSchema(propSchema);
        }
        out[key] = cleaned;
      } else if (key === "items") {
        out[key] = cleanSchema(value);
      } else {
        out[key] = value;
      }
    }
    return out;
  }
  return schema;
};

// OpenAI tool → Gemini functionDeclaration
const transformTool = (tool) => {
  if (tool.type !== "function" || !tool.function?.name) {
    throw new HttpError(`Unsupported tool: ${JSON.stringify(tool).slice(0, 200)}`, 400);
  }
  const { name, description, parameters } = tool.function;
  return {
    name,
    description: description ?? "",
    parameters: parameters ? cleanSchema(parameters) : { type: "object", properties: {} },
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
  const { system_instruction, contents } = await transformMessages(req.messages) || {};

  // 兜底防线：若本轮请求包含 functionCall 但全部缺少 thought_signature
  // （说明客户端无法回传签名），且用户未显式配置 thinking，
  // 则自动关闭思考，避免 Gemini 报错导致工具调用中断。
  let needDisableThinking = false;
  if (Array.isArray(contents) && req.thinking === undefined) {
    const hasFunctionCall = contents.some(c =>
      Array.isArray(c.parts) && c.parts.some(p => p.functionCall)
    );
    const allCallsHaveSignature = contents.every(c =>
      !Array.isArray(c.parts) || c.parts.every(p =>
        !p.functionCall || !!p.functionCall.thought_signature
      )
    );
    needDisableThinking = hasFunctionCall && !allCallsHaveSignature;
  }

  const generationConfig = transformConfig(req);
  if (needDisableThinking) {
    generationConfig.thinkingConfig = { thinkingBudget: 0 };
  }

  return {
    ...(system_instruction && { system_instruction }),
    ...(contents && { contents }),
    safetySettings,
    ...(geminiTools && { tools: geminiTools }),
    ...(toolConfig && { toolConfig }),
    generationConfig,
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

// === thought_signature 编解码 ===
// 背景：Gemini 思考模型的 functionCall part 必须携带 thought_signature，
// 但 OpenAI 格式的 tool_calls 没有标准字段，部分客户端会丢弃自定义字段。
// 解决：把签名编码进 tool_call 的 id（标准字段，客户端必然原样保留），
// 请求方向再从 id 解码还原。自定义字段 thought_signature 作为双保险同时附加。
const SIG_ID_PREFIX = "call_s_";

// 有签名时生成 "call_s_<base64url(sig)>" 的 id，无签名时正常随机 id
const encodeSignatureToId = (sig) => {
  if (!sig) return generateToolCallId();
  return SIG_ID_PREFIX + Buffer.from(String(sig), "utf8").toString("base64url").replace(/=+$/, "");
};

// 从 id 解码签名（仅识别 call_s_ 前缀的编码 id，普通随机 id 返回 undefined）
const decodeSignatureFromId = (id) => {
  if (typeof id !== "string" || !id.startsWith(SIG_ID_PREFIX)) { return; }
  const b64 = id.slice(SIG_ID_PREFIX.length);
  try {
    const decoded = Buffer.from(b64, "base64url").toString("utf8");
    return decoded.length > 0 ? decoded : undefined;
  } catch {
    return;
  }
};

// 获取 tool_call 上的 thought_signature：优先自定义字段，其次从 id 解码
const getThoughtSignature = (tc) => {
  if (!tc) { return; }
  if (tc.thought_signature) { return tc.thought_signature; }
  return decodeSignatureFromId(tc.id);
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
    message.tool_calls = callParts.map(p => {
      const sig = p.functionCall.thought_signature;
      const tc = {
        id: encodeSignatureToId(sig),
        type: "function",
        function: {
          name: p.functionCall.name,
          arguments: JSON.stringify(p.functionCall.args ?? {}),
        },
      };
      // 双保险：自定义字段 + id 编码
      if (sig) {
        tc.thought_signature = sig;
      }
      return tc;
    });
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
    delta.tool_calls = callParts.map(p => {
      const sig = p.functionCall.thought_signature;
      const tc = {
        index: this.toolCallIndex++,
        id: encodeSignatureToId(sig),
        type: "function",
        function: {
          name: p.functionCall.name,
          arguments: JSON.stringify(p.functionCall.args ?? {}),
        },
      };
      // 双保险：自定义字段 + id 编码
      if (sig) {
        tc.thought_signature = sig;
      }
      return tc;
    });
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
  transformTool,
  transformToolChoice,
  transformMessages,
  transformCandidates,
  transformConfig,
  transformRequest,
  generateToolCallId,
  cleanSchema,
  encodeSignatureToId,
  decodeSignatureFromId,
  getThoughtSignature,
};
