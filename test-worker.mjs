// worker.mjs 增强功能测试（不调用真实 API，纯转换逻辑验证）
import assert from "node:assert/strict";
import {
  transformTools,
  transformToolChoice,
  transformMessages,
  transformCandidates,
  transformRequest,
  generateToolCallId,
} from "./src/api_proxy/worker.mjs";

let passed = 0;
const ok = (name) => { passed++; console.log(`  ✅ ${name}`); };

console.log("1️⃣ tools 转换（OpenAI → Gemini functionDeclarations）");
{
  const tools = [{
    type: "function",
    function: {
      name: "get_weather",
      description: "获取天气",
      parameters: {
        type: "object",
        properties: { location: { type: "string" } },
        required: ["location"],
      },
    },
  }];
  const out = transformTools(tools);
  assert.deepEqual(out, [{
    functionDeclarations: [{
      name: "get_weather",
      description: "获取天气",
      parameters: { type: "object", properties: { location: { type: "string" } }, required: ["location"] },
    }],
  }]);
  ok("tools 数组正确映射为 functionDeclarations");
  assert.equal(transformTools([]), undefined);
  assert.equal(transformTools(undefined), undefined);
  ok("空/缺失 tools 返回 undefined");
}

console.log("2️⃣ tool_choice 转换（→ toolConfig）");
{
  assert.deepEqual(transformToolChoice("none"), { functionCallingConfig: { mode: "NONE" } });
  assert.deepEqual(transformToolChoice("required"), { functionCallingConfig: { mode: "ANY" } });
  assert.equal(transformToolChoice("auto"), undefined);
  assert.equal(transformToolChoice(undefined), undefined);
  assert.deepEqual(transformToolChoice({ type: "function", function: { name: "get_weather" } }),
    { functionCallingConfig: { mode: "ANY", allowedFunctionNames: ["get_weather"] } });
  ok("auto/none/required/指定函数 全部正确");
}

console.log("3️⃣ 多轮工具调用消息链路（assistant tool_calls + tool 结果）");
{
  const messages = [
    { role: "system", content: "You are a helpful assistant." },
    { role: "user", content: "北京天气怎么样？" },
    {
      role: "assistant",
      content: null,
      tool_calls: [{
        id: "call_123",
        type: "function",
        function: { name: "get_weather", arguments: "{\"location\":\"北京\"}" },
      }],
    },
    { role: "tool", tool_call_id: "call_123", content: "{\"weather\":\"sunny\",\"temp\":30}" },
    { role: "user", content: "好的谢谢" },
  ];
  const { system_instruction, contents } = await transformMessages(messages);
  assert.deepEqual(system_instruction, { parts: [{ text: "You are a helpful assistant." }] });
  assert.equal(contents.length, 4);
  assert.equal(contents[0].role, "user");
  assert.equal(contents[0].parts[0].text, "北京天气怎么样？");
  assert.equal(contents[1].role, "model");
  assert.deepEqual(contents[1].parts, [{ functionCall: { name: "get_weather", args: { location: "北京" } } }]);
  assert.equal(contents[2].role, "user");
  assert.deepEqual(contents[2].parts, [{ functionResponse: { name: "get_weather", response: { weather: "sunny", temp: 30 } } }]);
  assert.equal(contents[3].role, "user");
  // 原 messages 未被修改（无副作用）
  assert.equal(messages[2].role, "assistant");
  assert.equal(messages[0].role, "system");
  ok("tool_call_id→函数名映射、functionResponse 构造、无副作用 全部正确");
}

console.log("4️⃣ Gemini 响应 → OpenAI 格式（functionCall → tool_calls）");
{
  const cand = {
    index: 0,
    content: { parts: [{ functionCall: { name: "get_weather", args: { location: "北京" } } }] },
    finishReason: "STOP",
  };
  const out = transformCandidates("message", cand);
  assert.equal(out.message.content, null);
  assert.equal(out.message.tool_calls.length, 1);
  assert.equal(out.message.tool_calls[0].type, "function");
  assert.equal(out.message.tool_calls[0].function.name, "get_weather");
  assert.equal(out.message.tool_calls[0].function.arguments, "{\"location\":\"北京\"}");
  assert.ok(out.message.tool_calls[0].id.startsWith("call_"));
  assert.equal(out.finish_reason, "tool_calls");
  ok("functionCall 正确转为 tool_calls，finish_reason=tool_calls");
}

console.log("5️⃣ 并行函数调用（一个响应多个 functionCall）");
{
  const cand = {
    index: 0,
    content: {
      parts: [
        { text: "我来查一下：" },
        { functionCall: { name: "get_weather", args: { location: "北京" } } },
        { functionCall: { name: "get_weather", args: { location: "上海" } } },
      ],
    },
    finishReason: "STOP",
  };
  const out = transformCandidates("message", cand);
  assert.equal(out.message.content, "我来查一下：");
  assert.equal(out.message.tool_calls.length, 2);
  assert.equal(out.message.tool_calls[0].function.arguments, "{\"location\":\"北京\"}");
  assert.equal(out.message.tool_calls[1].function.arguments, "{\"location\":\"上海\"}");
  assert.notEqual(out.message.tool_calls[0].id, out.message.tool_calls[1].id);
  assert.equal(out.finish_reason, "tool_calls");
  ok("文本+多函数调用共存正确");
}

console.log("6️⃣ 纯文本响应不受影响");
{
  const out = transformCandidates("message", {
    index: 0,
    content: { parts: [{ text: "你好" }] },
    finishReason: "STOP",
  });
  assert.equal(out.message.content, "你好");
  assert.equal(out.message.tool_calls, undefined);
  assert.equal(out.finish_reason, "stop");
  ok("纯文本响应与原始行为一致");
}

console.log("7️⃣ 完整请求转换（tools + tool_choice + 生成参数 + thinking）");
{
  const req = {
    model: "gemini-3-flash",
    messages: [
      { role: "system", content: "助手" },
      { role: "user", content: "查天气" },
    ],
    tools: [{ type: "function", function: { name: "get_weather", description: "天气", parameters: { type: "object", properties: {} } } }],
    tool_choice: "auto",
    temperature: 0.7,
    max_tokens: 100,
    stream: true,
    thinking: { thinkingBudget: 0 },
  };
  const out = await transformRequest(req);
  assert.equal(out.tools[0].functionDeclarations[0].name, "get_weather");
  assert.equal(out.toolConfig, undefined); // auto 不传
  assert.equal(out.generationConfig.temperature, 0.7);
  assert.equal(out.generationConfig.maxOutputTokens, 100);
  assert.deepEqual(out.generationConfig.thinkingConfig, { thinkingBudget: 0 });
  assert.equal(out.system_instruction.parts[0].text, "助手");
  assert.equal(out.contents[0].parts[0].text, "查天气");
  ok("完整请求转换正确");
}

console.log("8️⃣ 旧版 functions 参数兼容 + thinking 数字简写");
{
  const out = await transformRequest({
    messages: [{ role: "user", content: "hi" }],
    functions: [{ name: "f1", description: "d", parameters: { type: "object", properties: {} } }],
  });
  assert.equal(out.tools[0].functionDeclarations[0].name, "f1");
  const out2 = await transformRequest({
    messages: [{ role: "user", content: "hi" }],
    thinking: 0,
  });
  assert.deepEqual(out2.generationConfig.thinkingConfig, { thinkingBudget: 0 });
  ok("functions 兼容与 thinking 数字简写正确");
}

console.log("9️⃣ tool_call_id 生成");
{
  const a = generateToolCallId();
  const b = generateToolCallId();
  assert.ok(a.startsWith("call_") && a.length === 29);
  assert.notEqual(a, b);
  ok("tool_call id 格式与唯一性正确");
}

console.log(`\n🎉 全部 ${passed} 组测试通过！`);
