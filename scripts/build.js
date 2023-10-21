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
  target: 'esnext',
  format: 'esm',
  platform: 'neutral',
  packages: 'external',
})
