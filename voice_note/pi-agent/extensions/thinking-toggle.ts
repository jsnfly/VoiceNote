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

	// DEBUG: log raw provider requests to stderr (visible in chat.py pi stderr logs)
	pi.on("before_provider_request", async (event: BeforeProviderRequestEvent) => {
		console.error("=== PROVIDER REQUEST ===");
		console.error(JSON.stringify(event.payload, null, 2));
	});

	// DEBUG: log response stream events that contain thinking/reasoning content
	pi.on("message_update", async (event: MessageUpdateEvent) => {
		const e = event.assistantMessageEvent;
		if (e.type === "thinking_delta" || e.type === "thinking_start" || e.type === "thinking_end") {
			console.error(`=== RESPONSE ${e.type} ===`);
			console.error(JSON.stringify(e, null, 2));
		}
	});
}
