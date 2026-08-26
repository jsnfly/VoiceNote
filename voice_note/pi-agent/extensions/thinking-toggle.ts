import type { ExtensionAPI, AgentToolResult, BeforeProviderRequestEvent, MessageUpdateEvent } from "@earendil-works/pi-coding-agent";
import { Type, type Static } from "typebox";

const LEVELS = ["off", "minimal", "low", "medium", "high"] as const;

const Params = Type.Object({
	level: Type.Optional(Type.String({ description: `Thinking level: ${LEVELS.join(", ")}` })),
});

export default function (pi: ExtensionAPI) {
	pi.registerTool({
		name: "set_thinking",
		label: "Set Thinking",
		description: "Get or set the thinking level. Omit level to read current value. Set to 'off' to disable; any other level enables thinking (binary on/off on this model).",
		promptSnippet: "set_thinking(level?) — get or set thinking mode",
		promptGuidelines: [
			"Use set_thinking when the user asks to enable/disable thinking or what mode is active.",
		],
		parameters: Params,

		async execute(_toolCallId: string, params: Static<typeof Params>): Promise<AgentToolResult> {
			if (!params.level) {
				return { content: [{ type: "text", text: `Current thinking level: ${pi.getThinkingLevel()}` }] };
			}

			if (!(LEVELS as readonly string[]).includes(params.level)) {
				return {
					content: [{ type: "text", text: `Invalid level "${params.level}". Valid: ${LEVELS.join(", ")}` }],
					isError: true,
				};
			}

			const prev = pi.getThinkingLevel();
			pi.setThinkingLevel(params.level);
			return { content: [{ type: "text", text: `Thinking: ${prev} → ${params.level}` }] };
		},
	});

	// Forcefully apply the current thinking level to every outgoing provider request.
	//
	// WHY THIS IS NEEDED:
	// pi-coding-agent's SDK builds the Agent without wiring `prepareNextTurn`
	// (see pi-coding-agent dist/core/sdk.js: `new Agent({...})` omits it). As a
	// result, pi-agent-core's agent loop snapshots `config.reasoning` once at the
	// start of `runPromptMessages` (pi-agent-core/dist/agent.js:282) and does NOT
	// refresh it between turns — the refresh block at
	// pi-agent-core/dist/agent-loop.js:132-144 is skipped because
	// `config.prepareNextTurn` is undefined. So a tool that calls
	// `pi.setThinkingLevel(level)` mid-run updates `agent.state.thinkingLevel`
	// immediately, but the agent's OWN next assistant turn in the same run still
	// uses the OLD `reasoning`/`reasoningEffort` value. The new level only takes
	// effect on the NEXT user prompt (a fresh `runPromptMessages`). This one-turn
	// lag is the "doesn't work reliably" symptom.
	//
	// `before_provider_request` fires on every provider request with the FINAL
	// built payload (post-buildParams, including `chat_template_kwargs`), and its
	// return value REPLACES what gets sent. So we re-derive `enable_thinking`
	// from the live `pi.getThinkingLevel()` here, bypassing the frozen
	// `config.reasoning`. This makes a tool-invoked toggle take effect on the
	// very next assistant turn.
	//
	// PROPER UPSTREAM FIX (not yet implemented as of pi 0.80.6 / main):
	// wire `prepareNextTurn` in sdk.ts when constructing `new Agent({...})`,
	// returning `{ thinkingLevel: this.thinkingLevel }`. The agent loop would
	// then refresh `config.reasoning` each turn (agent-loop.js:138-142), exactly
	// as the AgentHarness path already does (agent-harness.js:339,375). Until
	// that lands upstream, this payload override is the reliable workaround.
	//
	// NOTE: for `thinkingFormat: "qwen-chat-template"`, pi-ai maps any non-"off"
	// level to `enable_thinking: true` (binary). Gemma 4's chat template honors
	// `enable_thinking` (injects/suppresses <|think|>), confirmed via GET /props.
	// `preserve_thinking` is NOT a recognized llama.cpp kwarg and is dropped here.
	pi.on("before_provider_request", async (event: BeforeProviderRequestEvent) => {
		const payload = event.payload as Record<string, unknown>;
		const level = pi.getThinkingLevel();
		payload.chat_template_kwargs = { enable_thinking: level !== "off" };

		console.error("=== PROVIDER REQUEST ===");
		const { tools: _tools, ...logPayload } = payload;
		console.error(`thinking=${level} | ${JSON.stringify(logPayload)}`);
		return payload;
	});

	// DEBUG: log thinking-related response stream events
	pi.on("message_update", async (event: MessageUpdateEvent) => {
		const e = event.assistantMessageEvent;
		if (e.type === "thinking_start" || e.type === "thinking_end") {
			console.error(`=== RESPONSE ${e.type} ===`);
		}
	});
}
