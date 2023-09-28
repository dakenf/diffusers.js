import React, { useEffect, useRef, useState } from 'react'
import './App.css';
import { ProgressCallback, ProgressCallbackPayload, StableDiffusionPipeline } from './lib/StableDiffusionPipeline'
import { debounce } from 'lodash'
import CssBaseline from '@mui/material/CssBaseline';
import Box from '@mui/material/Box';
import Container from '@mui/material/Container';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import Alert from '@mui/material/Alert';
import Stack from '@mui/material/Stack';
import Grid from '@mui/material/Grid';
import TextField from '@mui/material/TextField';
import Button from '@mui/material/Button';
import List from '@mui/material/List';
import ListItem from '@mui/material/ListItem';
import ListItemButton from '@mui/material/ListItemButton';
import ListItemIcon from '@mui/material/ListItemIcon';
import ListItemText from '@mui/material/ListItemText';
import Divider from '@mui/material/Divider';
import { Checkbox } from '@mui/material';
import { FormControlLabel } from '@mui/material';

const darkTheme = createTheme({
  palette: {
    mode: 'dark',
  },
});
// async function inference () {
//   try {
//
//     // const pipe = await StableDiffusionPipeline.fromPretrained('webgl', '/aislamov/sd2_1base');
//     // @ts-ignore
//     window.pipe = pipe
//     // const result = pipe.run("a horse", "", 1, 7.5, 30)
//     // console.log(result);
//   } catch (e) {
//     console.error(e)
//   }
// }

// const debouncedInference = debounce(inference, 1000)

function FaqItem (props: { question: string, answer:string }){
  return (
    <>
      <ListItem>
        <ListItemText primary={'Q: ' + props.question} />
      </ListItem>
      <ListItem>
        <ListItemText primary={'A: ' + props.answer} />
      </ListItem>
      <Divider/>
    </>
  )
}

function FAQ () {
  return (
    <Box sx={{ width: '100%', bgcolor: 'background.paper' }}>
      <List>
        <ListItem>
          <h2>FAQ</h2>
        </ListItem>
        <FaqItem question={'What if I get protobuf parsing failed error?'} answer={'Open DevTools, go to Application -> Storage and press "Clear site data".'} />
        <FaqItem question={'What if I get sbox_fatal_memory_exceeded?'} answer={"You don't have enough RAM to run SD. You can try reloading the tab or browser."} />
        <FaqItem question={'How did you make it possible?'} answer={'In order to run it, I had to port StableDiffusionPipeline from python to JS. Then patch onnxruntime and emscripten+binaryen (WebAssembly compiler toolchain) to support allocating and using >4GB memory. Once my pull requests get to release, anyone would be able to compile and run code that uses >4GB in the browser.'} />
        <FaqItem question={'Why is it so slow?'} answer={'It does not support multi-threading yet, so is using just one CPU core. There is no way to create 64 bit memory with SharedArrayBuffer through WebAssembly.Memory constructor. I’ve proposed a spec change to have “memory64” flag and after it’s accepted, i will patch V8 engine to support it.'} />
        <FaqItem question={'But it’s running on a GPU, right?'} answer={'Yes, but webgpu is onnxruntime is in early stage, so a lot of operations are not yet implemented. And data is constantly transferred from and to CPU through JS. Once the majority of operations will have their JS kernels, it will be way faster.'} />
        <FaqItem question={'Can I run it locally?'} answer={'Yes, this page’s code is available here: https://github.com/dakenf/stable-diffusion-webgpu-minimal'} />
        <FaqItem question={'Can I use your patched onnxruntime to run big LLMs with transformers.js?'} answer={'Yes, you can use this package but i don’t guarantee it will be working in all cases. This build is limited to 8GB of memory, so you can load up to ~4GB weights. Just use https://www.npmjs.com/package/@aislamov/onnxruntime-web64'} />
        <FaqItem question={'Are you going to make a pull request in onnxruntime repo?'} answer={'Yes. It will be my second one, i’ve added GPU acceleration to node.js binding earlier.'} />
      </List>
    </Box>
  )
}

function App() {
  const [modelState, setModelState] = useState<'none'|'loading'|'ready'|'inferencing'>('none');
  const [prompt, setPrompt] = useState('an astronaut riding a horse');
  const [negativePrompt, setNegativePrompt] = useState('');
  const [inferenceSteps, setInferenceSteps] = useState(3);
  const [guidanceScale, setGuidanceScale] = useState(7.5);
  const [seed, setSeed] = useState('');
  const [status, setStatus] = useState('Ready');
  const pipeline = useRef<StableDiffusionPipeline|null>(null);
  const [img2img, setImg2Img] = useState(false);
  const [inputImage, setInputImage] = useState<Float32Array>();
  const [strength, setStrength] = useState(0.8);
  useEffect(() => {
    if (typeof window !== 'undefined') {
      // debouncedInference()
    }
  }, [])

  const progressCallback = async (info: ProgressCallbackPayload) => {
    if (info.step) {
      setStatus(info.step)
    }
    if (info.images) {
      const canvas = document.getElementById('canvas') as HTMLCanvasElement
      if (canvas) {
        const data = await info.images[0].toImageData({ tensorLayout: 'NCWH', format: 'RGB' });
        canvas.getContext('2d')!.putImageData(data, 0, 0);
      }
    }
  }
  const loadModel = () => {
    if (!window.confirm('This will download approximately 3.5gb and use 8gb of your RAM. Are you sure want to continue?')) {
      return
    }
    setModelState('loading')
    StableDiffusionPipeline.fromPretrained('webgpu', 'aislamov/stable-diffusion-2-1-base-onnx', progressCallback)
      .then((p) => {
        pipeline.current = p
        setModelState('ready')
      })
      .catch(e => alert(e))
  }

  function getRgbData(d: Uint8ClampedArray) {
    let rgbData: any = [[], [], []]; // [r, g, b]
    // remove alpha and put into correct shape:
    for(let i = 0; i < d.length; i += 4) { 
        let x = (i/4) % 512;
        let y = Math.floor((i/4) / 512)
        if(!rgbData[0][y]) rgbData[0][y] = [];
        if(!rgbData[1][y]) rgbData[1][y] = [];
        if(!rgbData[2][y]) rgbData[2][y] = [];
        rgbData[0][y][x] = (d[i+0]/255) * 2 - 1;
        rgbData[1][y][x] = (d[i+1]/255) * 2 - 1;
        rgbData[2][y][x] = (d[i+2]/255) * 2 - 1;
    }
    rgbData = Float32Array.from(rgbData.flat().flat());
    return rgbData;
  }

  function uploadImage(e: any) {
    if(!e.target.files[0]) {
      // No image uploaded
      return;
    }

    const uploadedImage = new Image(512, 512); // resize image to 512, 512
    const reader = new FileReader();
    // On file read loadend
    reader.addEventListener('loadend', function(file: any) {
      // On image load
      uploadedImage.addEventListener('load', function() {
        const imageCanvas = document.createElement('canvas');
        imageCanvas.width = uploadedImage.width;
        imageCanvas.height = uploadedImage.height;
        const imgCtx = imageCanvas.getContext('2d') as CanvasRenderingContext2D;
        imgCtx.drawImage(uploadedImage, 0, 0, uploadedImage.width, uploadedImage.height);
        const imageData = imgCtx.getImageData(0, 0, uploadedImage.width, uploadedImage.height).data;

        const rgb_array = getRgbData(imageData);
        setInputImage(rgb_array);
      });
      uploadedImage.src = file.target.result;
    });
    reader.readAsDataURL(e.target.files[0]);
  };

  const runInference = () => {
    if (!pipeline.current) {
      return
    }
    setModelState('inferencing')

    pipeline.current.run({
      prompt: prompt,
      negativePrompt: negativePrompt,
      numInferenceSteps: inferenceSteps,
      guidanceScale: guidanceScale,
      seed: seed,
      width: 512,
      height: 512,
      runVaeOnEachStep: true,
      progressCallback,
      img2imgFlag: img2img,
      inputImage: inputImage,
      strength: strength
    }).then(images => {
      progressCallback({
        step: 'Done',
        images,
      })
      setModelState('ready')
    })
  }
  return (
    <ThemeProvider theme={darkTheme}>
      <CssBaseline enableColorScheme={true} />
      <Container>
        <Stack>
          <Alert severity="error">You need latest Chrome with "Experimental WebAssembly" and "Experimental WebAssembly JavaScript Promise Integration (JSPI)" flags enabled!</Alert>
        </Stack>
        <Box sx={{ bgcolor: '#282c34' }} pt={4} pl={3} pr={3} pb={4}>
          <Grid container spacing={2}>
            <Grid item xs={6}>
              <Stack spacing={2}>
                <TextField
                  label="Prompt"
                  variant="standard"
                  disabled={modelState != 'ready'}
                  onChange={(e) => setPrompt(e.target.value)}
                  value={prompt}
                />
                <TextField
                  label="Negative Prompt"
                  variant="standard"
                  disabled={modelState != 'ready'}
                  onChange={(e) => setNegativePrompt(e.target.value)}
                  value={negativePrompt}
                />
                <TextField
                  label="Number of inference steps (Because of PNDM Scheduler, it will be i+1)"
                  variant="standard"
                  type='number'
                  disabled={modelState != 'ready'}
                  onChange={(e) => setInferenceSteps(parseInt(e.target.value))}
                  value={inferenceSteps}
                />
                <TextField
                  label="Guidance Scale. Controls how similar the generated image will be to the prompt."
                  variant="standard"
                  type='number'
                  InputProps={{ inputProps: { min: 1, max: 20, step: 0.5 } }}
                  disabled={modelState != 'ready'}
                  onChange={(e) => setGuidanceScale(parseFloat(e.target.value))}
                  value={guidanceScale}
                />
                <TextField
                  label="Seed (Creates initial random noise)"
                  variant="standard"
                  disabled={modelState != 'ready'}
                  onChange={(e) => setSeed(e.target.value)}
                  value={seed}
                />
                <FormControlLabel 
                  label="Check if you want to use the Img2Img pipeline"
                  control={<Checkbox
                    disabled={modelState != 'ready'}
                    onChange={(e) => setImg2Img(e.target.checked)}
                    checked={img2img}
                  />}
                />
                <label htmlFor="upload_image">Upload Image for Img2Img Pipeline:</label>
                <TextField
                  id="upload_image"
                  inputProps={{accept:"image/*"}}
                  type={"file"}
                  disabled={!img2img}
                  onChange={(e) => uploadImage(e)}
                />
                <TextField
                  label="Strength (Noise to add to input image). Value ranges from 0 to 1"
                  variant="standard"
                  type='number'
                  InputProps={{ inputProps: { min: 0, max: 1, step: 0.1 } }}
                  disabled={!img2img}
                  onChange={(e) => setStrength(parseFloat(e.target.value))}
                  value={strength}
                />
                <p>Each step takes about 1 minute + ~10sec to run VAE decoder to generate image. Having DevTools open will slow everything down to about 2x.
                  UNET runs only on CPU (it's 10% faster and does not give correct results on GPU), so will hang the browser tab.</p>
                <p>Minimum number of steps to get an acceptable result is 20. However, 3 would be fine for demo purposes.</p>
                <p>Model files will be cached and you won't need to download them each time.</p>
                <Button variant="outlined" onClick={loadModel} disabled={modelState != 'none'}>Load model</Button>
                <Button variant="outlined" onClick={runInference} disabled={modelState != 'ready'}>Run</Button>
                <p>{status}</p>
                <p><a href={'https://github.com/dakenf'}>Follow me on GitHub</a></p>
              </Stack>

            </Grid>
            <Grid item xs={6}>
              <canvas id={'canvas'} width={512} height={512} style={{ border: '1px dashed #ccc'}} />
            </Grid>
          </Grid>
        </Box>
        <Divider/>
        <FAQ />
      </Container>
    </ThemeProvider>
  );
}

export default App;
