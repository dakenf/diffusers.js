import * as esbuild from 'esbuild'

await esbuild.build({
  entryPoints: ['src/index.ts'],
  bundle: true,
  outfile: 'dist/index.js',
  platform: 'browser',
  packages: 'external',
})

await esbuild.build({
  entryPoints: ['src/index.ts'],
  bundle: true,
  outfile: 'dist/index.esm.js',
  platform: 'browser',
  format: 'esm',
  packages: 'external',
})

await esbuild.build({
  entryPoints: ['src/index-node.ts'],
  bundle: true,
  outfile: 'dist/index-node.esm.js',
  target: 'esnext',
  format: 'esm',
  platform: 'node',
  packages: 'external',
})

await esbuild.build({
  entryPoints: ['src/index-node.ts'],
  bundle: true,
  outfile: 'dist/index-node.cjs',
  target: 'esnext',
  format: 'cjs',
  platform: 'node',
  packages: 'external',
})
