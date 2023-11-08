# diffusers.js library for running diffusion models on GPU/WebGPU

See demo here https://islamov.ai/diffusers.js/

## Installation

```bash
npm i @aislamov/diffusers.js
```

## Usage

Browser (see examples/react)
```js
import { DiffusionPipeline } from '@aislamov/diffusers.js'

const pipe = DiffusionPipeline.fromPretrained('aislamov/stable-diffusion-2-1-base-onnx')
const images = pipe.run({
  prompt: "an astronaut running a horse",
  numInferenceSteps: 30,
})

const canvas = document.getElementById('canvas')
const data = await images[0].toImageData({ tensorLayout: 'NCWH', format: 'RGB' });
canvas.getContext('2d').putImageData(data, 0, 0);
```

Node.js (see examples/node)
```js
import { DiffusionPipeline } from '@aislamov/diffusers.js'
import { PNG } from 'pngjs'

const pipe = DiffusionPipeline.fromPretrained('aislamov/stable-diffusion-2-1-base-onnx')
const images = pipe.run({
  prompt: "an astronaut running a horse",
  numInferenceSteps: 30,
})

const data = await images[0].mul(255).round().clipByValue(0, 255).transpose(0, 2, 3, 1)

const p = new PNG({ width: 512, height: 512, inputColorType: 2 })
p.data = Buffer.from(data.data)
p.pack().pipe(fs.createWriteStream('output.png')).on('finish', () => {
  console.log('Image saved as output.png');
})
```

'aislamov/stable-diffusion-2-1-base-onnx' model is optimized for GPU and will fail to load without CUDA/DML/WebGPU support. Please use 'cpu' revision on a machine without GPU.
```js
const pipe = DiffusionPipeline.fromPretrained('aislamov/stable-diffusion-2-1-base-onnx', { revision: 'cpu' })
```

## Running examples
Browser/React
```bash
npm install && npm run build
cd examples/react && npm install
npm run start
```

Node
```bash
npm install && npm run build
cd examples/node && npm install
node src/txt2img.mjs --prompt "a dog" --rev cpu
```

## Node.js GPU support

DML and CUDA support for node.js is already merged to onnxruntime https://github.com/microsoft/onnxruntime/pull/16050 but it is not released in current onnxruntime-node package. I will build my own version if they will not update it in the next release.

## How does it work

It uses my own modified build of onnx runtime for web with 64bit and other changes. You can see list of contributions here https://github.com/search?q=is%3Apr%20author%3Adakenf%20&type=pullrequests

I'm trying to get all changes to be merged to official version.

Also, i've had to do a lot of fixes to emscripten like this https://github.com/emscripten-core/emscripten/pull/19737, WebAssembly spec and V8 engine like [this](https://chromium-review.googlesource.com/c/v8/v8/+/4742982) and [this](https://chromium-review.googlesource.com/c/v8/v8/+/4775578) in order to make this all work  
