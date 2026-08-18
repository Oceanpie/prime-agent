import type { AgentEvent } from "@earendil-works/pi-agent-core";
import { type AssistantMessage, fauxAssistantMessage } from "@earendil-works/pi-ai";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { AgentSessionRuntime } from "../../../src/core/agent-session-runtime.js";
import { InProcessAgentConnection } from "../../../src/modes/agent-connection/in-process-agent-connection.js";
import { createHarness, type Harness } from "../harness.js";

function runtimeHostFor(session: unknown): AgentSessionRuntime {
	return {
		session,
		setRebindSession() {},
		setBeforeSessionInvalidate() {},
		async dispose() {},
	} as unknown as AgentSessionRuntime;
}

function provider401Message(): AssistantMessage {
	return {
		...fauxAssistantMessage("", {
			stopReason: "error",
			errorMessage: "401 Unauthorized: invalid API key",
		}),
		diagnostics: [
			{
				type: "provider_stream_failure",
				timestamp: Date.now(),
				details: { kind: "auth", status: 401 },
			},
		],
	};
}

function bareProvider401Message(): AssistantMessage {
	return fauxAssistantMessage("", {
		stopReason: "error",
		errorMessage: "401 status code (no body)",
	});
}

function provider500Message(): AssistantMessage {
	return {
		...fauxAssistantMessage("", {
			stopReason: "error",
			errorMessage: "500 Internal Server Error",
		}),
		diagnostics: [
			{
				type: "provider_stream_failure",
				timestamp: Date.now(),
				details: { kind: "server_error", status: 500 },
			},
		],
	};
}

const staleRecoveryModels = [
	{ id: "faux-1", name: "Faux One" },
	{ id: "faux-2", name: "Faux Two" },
];

function markHarnessProviderAuthStale(harness: Harness, provider: string): void {
	let marked = false;
	while (harness.session.modelRegistry.markProviderAuthStale(provider)) marked = true;
	expect(marked).toBe(true);
	expect(harness.session.modelRegistry.getProviderAuthStatus(provider)).toMatchObject({ source: "stale" });
}

function preserveFauxProviderDuringModelRefresh(harness: Harness): void {
	vi.spyOn(harness.session.modelRegistry, "refreshAvailableModels").mockImplementation(async () =>
		harness.session.modelRegistry.getAvailable(),
	);
}

describe("issue #4491 provider stale after repeated 401", () => {
	const harnesses: Harness[] = [];

	afterEach(() => {
		vi.restoreAllMocks();
		while (harnesses.length > 0) {
			harnesses.pop()?.cleanup();
		}
	});

	it("retries structured provider auth failures once, then marks current auth stale", async () => {
		const harness = await createHarness({
			settings: { retry: { enabled: true, maxRetries: 2, baseDelayMs: 1 } },
		});
		harnesses.push(harness);
		harness.setResponses([provider401Message(), provider401Message(), provider401Message()]);

		await harness.session.prompt("hello");

		expect(harness.faux.state.callCount).toBe(2);
		expect(harness.eventsOfType("auto_retry_start").map((event) => event.attempt)).toEqual([1]);
		expect(harness.eventsOfType("auto_retry_end").map((event) => event.success)).toEqual([false]);
		expect(harness.eventsOfType("auth_stale")).toHaveLength(1);

		const provider = harness.getModel().provider;
		expect(harness.authStorage.hasAuth(provider)).toBe(false);
		expect(harness.authStorage.getAuthStatus(provider)).toEqual({
			configured: false,
			source: "stale",
			label: "expired",
		});

		const assistantMessages = harness.session.messages.filter(
			(message): message is AssistantMessage => message.role === "assistant",
		);
		const finalAssistant = assistantMessages[assistantMessages.length - 1];
		expect(finalAssistant?.errorMessage).toContain("401 Unauthorized");
		expect(finalAssistant?.errorMessage).toContain("Run /login to update credentials.");
	});

	it("emits stale auth source tokens for daemon clients after bare 401 auth failures", async () => {
		const harness = await createHarness({
			provider: "prime-inference",
			settings: { retry: { enabled: true, maxRetries: 0, baseDelayMs: 1 } },
		});
		harnesses.push(harness);
		harness.setResponses([bareProvider401Message()]);

		await harness.session.prompt("hello");

		const authStaleEvents = harness.eventsOfType("auth_stale");
		expect(authStaleEvents).toHaveLength(1);
		expect(authStaleEvents[0]?.provider).toBe("prime-inference");
		expect(authStaleEvents[0]?.sourceTokens).toMatchObject([
			{
				provider: "prime-inference",
				source: "runtime",
			},
		]);
		expect(harness.authStorage.getAuthStatus("prime-inference")).toEqual({
			configured: false,
			source: "stale",
			label: "expired",
		});
	});

	it("classifies bare status-code auth failures before login guidance is appended", async () => {
		const harness = await createHarness({
			provider: "prime-inference",
			settings: { retry: { enabled: true, maxRetries: 1, baseDelayMs: 1 } },
		});
		harnesses.push(harness);
		const message = bareProvider401Message();
		const event = { type: "agent_end", messages: [message] } as AgentEvent;
		const session = harness.session as unknown as {
			_createRetryPromiseForAgentEnd(event: AgentEvent): void;
		};

		session._createRetryPromiseForAgentEnd(event);

		expect(harness.session.isRetrying).toBe(true);
		harness.session.abortRetry();
	});

	it("creates retry promises for exhausted structured auth failures so cleanup is awaited", async () => {
		const harness = await createHarness({
			settings: { retry: { enabled: true, maxRetries: 2, baseDelayMs: 1 } },
		});
		harnesses.push(harness);
		const event = { type: "agent_end", messages: [provider401Message()] } as AgentEvent;
		const session = harness.session as unknown as {
			_retryAttempt: number;
			_createRetryPromiseForAgentEnd(event: AgentEvent): void;
		};
		session._retryAttempt = 1;

		session._createRetryPromiseForAgentEnd(event);

		expect(harness.session.isRetrying).toBe(true);
		harness.session.abortRetry();
	});

	it("marks captured auth failures stale when retry backoff is cancelled", async () => {
		const harness = await createHarness({
			settings: { retry: { enabled: true, maxRetries: 3, baseDelayMs: 100 } },
		});
		harnesses.push(harness);
		harness.setResponses([provider401Message(), provider401Message()]);

		const sawRetryStart = new Promise<void>((resolve) => {
			const unsubscribe = harness.session.subscribe((event) => {
				if (event.type === "auto_retry_start") {
					unsubscribe();
					resolve();
				}
			});
		});

		const promptPromise = harness.session.prompt("hello");
		await sawRetryStart;
		harness.session.abortRetry();
		await promptPromise;

		expect(harness.faux.state.callCount).toBe(1);
		expect(harness.eventsOfType("auth_stale")).toHaveLength(1);
		expect(harness.eventsOfType("auto_retry_end").map((event) => event.finalError)).toContain("Retry cancelled");
		expect(harness.authStorage.hasAuth(harness.getModel().provider)).toBe(false);
		await expect(harness.authStorage.getApiKey(harness.getModel().provider)).resolves.toBeUndefined();
	});

	it("marks each failed auth source stale when credentials change during retry backoff", async () => {
		const harness = await createHarness({
			settings: { retry: { enabled: true, maxRetries: 1, baseDelayMs: 5 } },
		});
		harnesses.push(harness);
		harness.setResponses([provider401Message(), provider401Message()]);

		let changedCredentials = false;
		harness.session.subscribe((event) => {
			if (event.type === "auto_retry_start" && !changedCredentials) {
				changedCredentials = true;
				harness.authStorage.setRuntimeApiKey(harness.getModel().provider, "fresh-key");
			}
		});

		await harness.session.prompt("hello");

		expect(changedCredentials).toBe(true);
		expect(harness.faux.state.callCount).toBe(2);
		expect(harness.authStorage.hasAuth(harness.getModel().provider)).toBe(false);
		await expect(harness.authStorage.getApiKey(harness.getModel().provider)).resolves.toBeUndefined();
		expect(harness.authStorage.getAuthStatus(harness.getModel().provider)).toEqual({
			configured: false,
			source: "stale",
			label: "expired",
		});
	});

	it("marks captured auth failures stale when the final retryable error is not auth", async () => {
		const harness = await createHarness({
			settings: { retry: { enabled: true, maxRetries: 2, baseDelayMs: 1 } },
		});
		harnesses.push(harness);
		harness.setResponses([provider401Message(), provider500Message(), provider500Message()]);

		await harness.session.prompt("hello");

		expect(harness.faux.state.callCount).toBe(3);
		expect(harness.eventsOfType("auto_retry_start").map((event) => event.attempt)).toEqual([1, 2]);
		expect(harness.eventsOfType("auto_retry_end").map((event) => event.success)).toEqual([false]);
		expect(harness.authStorage.hasAuth(harness.getModel().provider)).toBe(false);
		await expect(harness.authStorage.getApiKey(harness.getModel().provider)).resolves.toBeUndefined();

		const assistantMessages = harness.session.messages.filter(
			(message): message is AssistantMessage => message.role === "assistant",
		);
		const finalAssistant = assistantMessages[assistantMessages.length - 1];
		expect(finalAssistant?.errorMessage).toContain("500 Internal Server Error");
		expect(finalAssistant?.errorMessage).toContain("Run /login to update credentials.");
	});

	it("marks concrete auth failures stale when retry is disabled", async () => {
		const harness = await createHarness({
			settings: { retry: { enabled: false } },
		});
		harnesses.push(harness);
		harness.setResponses([provider401Message()]);

		await harness.session.prompt("hello");

		expect(harness.faux.state.callCount).toBe(1);
		expect(harness.eventsOfType("auto_retry_start")).toHaveLength(0);
		expect(harness.authStorage.hasAuth(harness.getModel().provider)).toBe(false);
		await expect(harness.authStorage.getApiKey(harness.getModel().provider)).resolves.toBeUndefined();
	});

	it("keeps auth stale after an explicit same-provider model selection", async () => {
		const harness = await createHarness({ models: staleRecoveryModels });
		harnesses.push(harness);
		const provider = harness.getModel().provider;
		const nextModel = harness.getModel("faux-2")!;
		markHarnessProviderAuthStale(harness, provider);
		preserveFauxProviderDuringModelRefresh(harness);

		const connection = new InProcessAgentConnection(runtimeHostFor(harness.session));
		await connection.setModel(provider, nextModel.id);

		expect(harness.session.model?.id).toBe(nextModel.id);
		expect(harness.session.modelRegistry.getProviderAuthStatus(provider)).toMatchObject({ source: "stale" });
		expect(harness.faux.state.callCount).toBe(0);
	});

	it("clears the stale auth source after the selected model succeeds", async () => {
		const harness = await createHarness({ models: staleRecoveryModels });
		harnesses.push(harness);
		const provider = harness.getModel().provider;
		const nextModel = harness.getModel("faux-2")!;
		markHarnessProviderAuthStale(harness, provider);
		preserveFauxProviderDuringModelRefresh(harness);
		harness.setResponses([fauxAssistantMessage("ok")]);

		const connection = new InProcessAgentConnection(runtimeHostFor(harness.session));
		await connection.setModel(provider, nextModel.id);
		await harness.session.prompt("hello");

		expect(harness.session.modelRegistry.getProviderAuthStatus(provider)).toMatchObject({ configured: true });
	});

	it("restores the stale auth source after the selected model fails", async () => {
		const harness = await createHarness({
			models: staleRecoveryModels,
			settings: { retry: { enabled: false } },
		});
		harnesses.push(harness);
		const provider = harness.getModel().provider;
		const nextModel = harness.getModel("faux-2")!;
		markHarnessProviderAuthStale(harness, provider);
		preserveFauxProviderDuringModelRefresh(harness);
		harness.setResponses([provider500Message()]);

		const connection = new InProcessAgentConnection(runtimeHostFor(harness.session));
		await connection.setModel(provider, nextModel.id);
		await harness.session.prompt("hello");

		expect(harness.session.modelRegistry.getProviderAuthStatus(provider)).toMatchObject({ source: "stale" });
	});

	it("keeps the original failed model gated until the selected model succeeds", async () => {
		const harness = await createHarness({ models: staleRecoveryModels });
		harnesses.push(harness);
		const originalModel = harness.getModel();
		const nextModel = harness.getModel("faux-2")!;
		markHarnessProviderAuthStale(harness, originalModel.provider);
		preserveFauxProviderDuringModelRefresh(harness);

		const connection = new InProcessAgentConnection(runtimeHostFor(harness.session));
		await connection.setModel(originalModel.provider, nextModel.id);
		await expect(connection.setModel(originalModel.provider, originalModel.id)).rejects.toThrow();

		expect(harness.session.modelRegistry.getProviderAuthStatus(originalModel.provider)).toMatchObject({
			source: "stale",
		});
	});

	it("keeps direct internal model changes on the standard auth gate", async () => {
		const harness = await createHarness({ models: staleRecoveryModels });
		harnesses.push(harness);
		const nextModel = harness.getModel("faux-2")!;
		markHarnessProviderAuthStale(harness, nextModel.provider);

		await expect(harness.session.setModel(nextModel)).rejects.toThrow(
			`No API key for ${nextModel.provider}/${nextModel.id}`,
		);
	});

	it("resolves retry state for auth failures surfaced only on agent_end", async () => {
		const harness = await createHarness({
			settings: { retry: { enabled: true, maxRetries: 0, baseDelayMs: 1 } },
		});
		harnesses.push(harness);
		const message = provider401Message();
		const event = { type: "agent_end", messages: [message] } as AgentEvent;
		const session = harness.session as unknown as {
			_createRetryPromiseForAgentEnd(event: AgentEvent): void;
			_processAgentEvent(event: AgentEvent): Promise<void>;
		};

		session._createRetryPromiseForAgentEnd(event);
		await session._processAgentEvent(event);

		expect(harness.session.isRetrying).toBe(false);
		expect(harness.eventsOfType("auto_retry_end").map((retryEvent) => retryEvent.success)).toEqual([false]);
		expect(harness.authStorage.hasAuth(harness.getModel().provider)).toBe(false);
		expect(message.errorMessage).toContain("Run /login to update credentials.");
	});
});
