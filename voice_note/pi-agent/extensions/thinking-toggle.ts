import type { ExtensionAPI, AgentToolResult, BeforeProviderRequestEvent, MessageUpdateEvent } from "@earendil-works/pi-coding-agent";
import { Type, type Static } from "typebox";

const LEVELS = ["off", "on"] as const;
// pi's internal thinking level used when the mode is "on".
// "medium" is sent by pi-ai as reasoning_effort: "medium", which the llama.cpp
// chat templates accept (they reject pi's other level names like "high").
const ON_LEVEL = "medium";

const Params = Type.Object({
	level: Type.Optional(Type.String({ description: `Thinking mode: ${LEVELS.join(" or ")}` })),
});

function display(level: string): "on" | "off" {
	return level === "off" ? "off" : "on";
}

export default function (pi: ExtensionAPI) {
	pi.registerTool({
		name: "set_thinking",
		label: "Set Thinking",
		description: "Turn thinking mode on or off. Omit level to read the current value.",
		promptSnippet: "set_thinking('on'|'off') — turn thinking mode on or off",
		promptGuidelines: [
			"Use set_thinking when the user asks to enable/disable thinking or asks what mode is active.",
			"Always change the thinking mode by calling this tool. Never claim to have changed it without calling it.",
		],
		parameters: Params,

		async execute(_toolCallId: string, params: Static<typeof Params>): Promise<AgentToolResult> {
			const current = display(pi.getThinkingLevel());

			if (!params.level) {
				return { content: [{ type: "text", text: `Thinking mode is ${current}.` }] };
			}

			const level = params.level.trim().toLowerCase();
			if (level !== "off" && level !== "on") {
				return {
					content: [{ type: "text", text: `Invalid level "${params.level}". Valid: off, on` }],
					isError: true,
				};
			}

			pi.setThinkingLevel(level === "on" ? ON_LEVEL : "off");
			return { content: [{ type: "text", text: `Thinking mode: ${current} → ${level}` }] };
		},
	});

	// Forcefully apply the current thinking mode to every outgoing provider request.
	//
	// WHY THIS IS STILL NEEDED (verified against pi 0.85.1, current latest):
	// pi-coding-agent's SDK builds the Agent without wiring `prepareNextTurn`
	// (dist/core/sdk.js: `new Agent({...})` omits it). pi-agent-core's agent loop
	// only refreshes `config.reasoning` via `config.prepareNextTurn?.()`, which is
	// therefore never invoked. So a tool that calls `pi.setThinkingLevel(level)`
	// mid-run updates `agent.state.thinkingLevel` immediately, but the agent's OWN
	// next assistant turn in the same run still uses the OLD reasoning value — the
	// new level would only take effect on the next user prompt. Since
	// `before_provider_request` fires on every provider request with the FINAL
	// built payload, and its return value REPLACES what gets sent, we re-derive
	// the thinking flag from the live `pi.getThinkingLevel()` here. This makes a
	// tool-invoked toggle take effect on the very next assistant turn.
	//
	// With "on", pi-ai additionally sends reasoning_effort (from ON_LEVEL), which
	// llama.cpp accepts. With "off", pi-ai sends no reasoning_effort at all.
	// `preserve_thinking` is NOT a recognized llama.cpp kwarg and is dropped here.
	pi.on("before_provider_request", async (event: BeforeProviderRequestEvent) => {
		const payload = event.payload as Record<string, unknown>;
		const level = pi.getThinkingLevel();
		payload.chat_template_kwargs = { enable_thinking: level !== "off" };

		if (process.env.PI_DEBUG_PAYLOAD === "1") {
			console.error("=== PROVIDER REQUEST ===");
			const { tools: _tools, ...logPayload } = payload;
			console.error(`thinking=${level} | ${JSON.stringify(logPayload)}`);
		}
		return payload;
	});

	// DEBUG: log thinking-related response stream events
	if (process.env.PI_DEBUG_PAYLOAD === "1") {
		pi.on("message_update", async (event: MessageUpdateEvent) => {
			const e = event.assistantMessageEvent;
			if (e.type === "thinking_start" || e.type === "thinking_end") {
				console.error(`=== RESPONSE ${e.type} ===`);
			}
		});
	}
}
