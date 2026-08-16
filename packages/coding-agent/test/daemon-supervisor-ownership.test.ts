import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { homedir, tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { DaemonSupervisor } from "../src/modes/daemon/daemon-supervisor.js";
import {
	acquireDaemonSupervisorOwnership,
	defaultDaemonSupervisorRegistryDir,
} from "../src/modes/daemon/daemon-supervisor-ownership.js";

type Ownership = Awaited<ReturnType<typeof acquireDaemonSupervisorOwnership>>;

interface OwnerRecord {
	token: string;
	generation: string;
	updatedAt: string;
	[key: string]: unknown;
}

const registryDirEnv = "PRIME_AGENT_INTERNAL_DAEMON_SUPERVISOR_REGISTRY_DIR";
const cleanupDirs: string[] = [];

afterEach(() => {
	while (cleanupDirs.length > 0) {
		const dir = cleanupDirs.pop();
		if (dir) rmSync(dir, { recursive: true, force: true });
	}
});

function createPaths(): {
	root: string;
	registryDir: string;
	socketPath: string;
	agentDir: string;
	descriptorDir: string;
} {
	const root = mkdtempSync(join(tmpdir(), "ownership-registry-"));
	cleanupDirs.push(root);
	const agentDir = join(root, "agent");
	mkdirSync(agentDir, { recursive: true });
	return {
		root,
		registryDir: join(root, "registry"),
		socketPath: join(root, "daemon.sock"),
		agentDir,
		descriptorDir: join(root, "workers"),
	};
}

async function acquire(paths: ReturnType<typeof createPaths>, generation = "registry-owner"): Promise<Ownership> {
	return acquireDaemonSupervisorOwnership({
		agentDir: paths.agentDir,
		appVersion: "test",
		descriptorDir: paths.descriptorDir,
		generation,
		registryDir: paths.registryDir,
		socketPath: paths.socketPath,
	});
}

function ownerDir(paths: ReturnType<typeof createPaths>, generation: string): string {
	return join(paths.registryDir, `${generation}.owner`);
}

function readJson(path: string): OwnerRecord {
	return JSON.parse(readFileSync(path, "utf8")) as OwnerRecord;
}

describe("daemon supervisor ownership registry", () => {
	it("derives the default registry from the durable per-user directory, env override first", () => {
		// Never touch the real ~/.prime from tests: probe the derivation only.
		expect(defaultDaemonSupervisorRegistryDir({})).toBe(join(homedir(), ".prime", "supervisor-owners"));
		expect(defaultDaemonSupervisorRegistryDir({ [registryDirEnv]: "/custom/registry" })).toBe("/custom/registry");
	});

	it("marks ownership lost when the record is mismatched or absent", async () => {
		const paths = createPaths();
		const mismatched = await acquire(paths, "mismatched-owner");
		const mismatchedPath = join(ownerDir(paths, "mismatched-owner"), "owner.json");
		const foreign = { ...readJson(mismatchedPath), token: "successor-token" };
		writeFileSync(mismatchedPath, `${JSON.stringify(foreign, null, 2)}\n`);

		await expect(mismatched.assertCurrent()).rejects.toMatchObject({
			code: "supervisor_generation_stale",
			name: "DaemonSupervisorOwnershipLostError",
		});
		expect(readJson(mismatchedPath).token).toBe("successor-token");
		await mismatched.release();
		rmSync(ownerDir(paths, "mismatched-owner"), { recursive: true, force: true });

		const reaped = await acquire(paths, "reaped-owner");
		const reapedDir = ownerDir(paths, "reaped-owner");
		rmSync(reapedDir, { recursive: true, force: true });
		await expect(reaped.assertCurrent()).rejects.toMatchObject({ code: "supervisor_generation_stale" });
		expect(existsSync(reapedDir)).toBe(false);
		await reaped.release();
	});

	it("disambiguates never-acquired from lost-on-disk ownership errors", async () => {
		const paths = createPaths();
		const supervisor = Object.assign(Object.create(DaemonSupervisor.prototype) as object, {
			ownership: undefined,
			generation: "unowned-generation",
			socketPath: paths.socketPath,
		});
		const assertCurrentOwnership = Reflect.get(supervisor, "assertCurrentOwnership") as () => Promise<void>;
		const neverAcquired = await assertCurrentOwnership
			.call(supervisor)
			.then(() => undefined)
			.catch((error: unknown) => error as Error & { code?: string });
		if (!neverAcquired) throw new Error("assertCurrentOwnership did not throw");
		expect(neverAcquired.code).toBe("supervisor_generation_stale");
		expect(neverAcquired.message).toContain("holds no registry ownership");
		expect(neverAcquired.message).toContain(paths.socketPath);
		expect(neverAcquired.message).toContain("sessions are preserved");

		const ownership = await acquire(paths);
		const ownerPath = join(ownerDir(paths, ownership.record.generation), "owner.json");
		const foreign = { ...readJson(ownerPath), token: "successor-token" };
		writeFileSync(ownerPath, `${JSON.stringify(foreign, null, 2)}\n`);
		const lostOnDisk = await ownership
			.assertCurrent()
			.then(() => undefined)
			.catch((error: unknown) => error as Error & { code?: string });
		if (!lostOnDisk) throw new Error("assertCurrent did not throw");
		expect(lostOnDisk.code).toBe("supervisor_generation_stale");
		expect(lostOnDisk.message).toContain("no longer owns its registry entry");
		expect(lostOnDisk.message).toContain(paths.socketPath);
		expect(lostOnDisk.message).toContain(paths.registryDir);
		expect(lostOnDisk.message).toContain("sessions are preserved");
		expect(lostOnDisk.message).not.toBe(neverAcquired.message);
		await ownership.release();
	});
});
