import minimist from 'minimist';
import { StableDiffusionPipeline } from '@aislamov/diffusers.js'
import fs from 'fs'
import { PNG } from 'pngjs'
import progress from 'cli-progress'

function parseCommandLineArgs() {
  const args = minimist(process.argv.slice(2));

  return {
    m: args.m || 'aislamov/stable-diffusion-2-1-base-onnx',
    prompt: args.prompt || 'an astronaut riding a horse',
    negativePrompt: args.negativePrompt || '',
    rev: args.rev,
    version: args.version || 2,
    steps: args.steps || 30,
  }
}

async function main() {
  const args = parseCommandLineArgs();
  const pipe = await StableDiffusionPipeline.fromPretrained(
    args.m,
    args.rev,
    async () => {}
  )

  const progressBar = new progress.SingleBar({
  }, progress.Presets.shades_classic)

  progressBar.start(args.steps + 1, 0)

  const images = await pipe.run({
    prompt: args.prompt,
    negativePrompt: args.negativePrompt,
    numInferenceSteps: args.steps,
    sdV1: args.version === 1,
    height: 512,
    width: 512,
    guidanceScale: 7.5,
    img2imgFlag: false,
    progressCallback: (progress) => {
      progressBar.update(progress.unetTimestep)
    },
  })
  progressBar.stop()
  const data = await images[0].mul(255).round().clipByValue(0, 255).transpose(0, 2, 3, 1)

  const p = new PNG({ width: 512, height: 512, inputColorType: 2 })
  p.data = Buffer.from(data.data)
  p.pack().pipe(fs.createWriteStream('output.png')).on('finish', () => {
    console.log('Image saved as output.png');
  })
}

main();
