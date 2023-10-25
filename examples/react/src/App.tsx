import React, { useEffect, useRef, useState } from 'react'
import './App.css';
import {
  DiffusionPipeline,
  ProgressCallback,
  ProgressCallbackPayload,
  setModelCacheDir,
  StableDiffusionPipeline,
  StableDiffusionXLPipeline
} from '@aislamov/diffusers.js'
import CssBaseline from '@mui/material/CssBaseline';
import Box from '@mui/material/Box';
import Container from '@mui/material/Container';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import Stack from '@mui/material/Stack';
import Grid from '@mui/material/Grid';
import TextField from '@mui/material/TextField';
import Button from '@mui/material/Button';
import Divider from '@mui/material/Divider';
import { Checkbox } from '@mui/material';
import { FormControlLabel } from '@mui/material';
import { BrowserFeatures } from './components/BrowserFeatures'
import { FAQ } from './components/FAQ'
import { Tensor } from '@xenova/transformers'

const darkTheme = createTheme({
  palette: {
    mode: 'dark',
  },
});

function App() {
  const [modelState, setModelState] = useState<'none'|'loading'|'ready'|'inferencing'>('none');
  const [prompt, setPrompt] = useState('An astronaut riding a green horse');
  const [negativePrompt, setNegativePrompt] = useState('');
  const [inferenceSteps, setInferenceSteps] = useState(3);
  const [guidanceScale, setGuidanceScale] = useState(5);
  const [seed, setSeed] = useState('');
  const [status, setStatus] = useState('Ready');
  const pipeline = useRef<StableDiffusionXLPipeline|StableDiffusionPipeline|null>(null);
  const [img2img, setImg2Img] = useState(false);
  const [inputImage, setInputImage] = useState<Float32Array>();
  const [strength, setStrength] = useState(0.8);
  const [runVaeOnEachStep, setRunVaeOnEachStep] = useState(false);
  useEffect(() => {
    setModelCacheDir('models')
  }, [])

  const drawImage = async (image: Tensor) => {
    const canvas = document.getElementById('canvas') as HTMLCanvasElement
    if (canvas) {
      const data = await image.toImageData({ tensorLayout: 'NCWH', format: 'RGB' });
      canvas.getContext('2d')!.putImageData(data, 0, 0);
    }
  }

  const progressCallback = async (info: ProgressCallbackPayload) => {
    if (info.statusText) {
      setStatus(info.statusText)
    }

    if (info.images) {
      await drawImage(info.images[0])
    }
  }

  const loadModel = async () => {
    if (!window.confirm('This will download approximately 2.5gb, use about 5gb of your RAM and up to 12gb VRAM. Are you sure want to continue?')) {
      return
    }
    setModelState('loading')
    try {
      pipeline.current = await DiffusionPipeline.fromPretrained(
        'aislamov/stable-diffusion-2-1-base-onnx',
        {
          progressCallback
        }
      )
      setModelState('ready')
    } catch (e) {
      alert(e)
      console.error(e)
    }
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
  }

  const runInference = async () => {
    if (!pipeline.current) {
      return
    }
    setModelState('inferencing')

    const images = await pipeline.current.run({
      prompt: prompt,
      negativePrompt: negativePrompt,
      numInferenceSteps: inferenceSteps,
      guidanceScale: guidanceScale,
      seed: seed,
      width: 512,
      height: 512,
      runVaeOnEachStep,
      progressCallback,
      img2imgFlag: img2img,
      inputImage: inputImage,
      strength: strength
    })
    await drawImage(images[0])
    setModelState('ready')
  }

  return (
    <ThemeProvider theme={darkTheme}>
      <CssBaseline enableColorScheme={true} />
      <Container>
        <BrowserFeatures />
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
                <FormControlLabel
                  label="Check if you want to run VAE after each step"
                  control={<Checkbox
                    disabled={modelState != 'ready'}
                    onChange={(e) => setRunVaeOnEachStep(e.target.checked)}
                    checked={runVaeOnEachStep}
                  />}
                />
                <p>Press the button below to download StableDiffusion 2.1 base. It will be stored in your browser cache.</p>
                <p>All settings above will become editable once model is downloaded.</p>
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
