import { cp, mkdir, readdir } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const root = fileURLToPath(new URL('..', import.meta.url));
const source = path.join(root, '..', 'web');
const target = path.join(root, 'dist');
await mkdir(target, { recursive: true });
// Pages' advanced-mode worker must not become a downloadable static asset.
for (const entry of await readdir(source, { withFileTypes: true })) {
  if (entry.name.startsWith('_') || entry.name.startsWith('.')) continue;
  await cp(path.join(source, entry.name), path.join(target, entry.name), { recursive: true });
}
