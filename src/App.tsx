import React, { useEffect, useRef, useState } from 'react'
import logo from './logo.svg';
import { Tensor, InferenceSession } from "onnxruntime-web";
import './App.css';
import { ProgressCallback, StableDiffusionPipeline } from './lib/StableDiffusionPipeline'
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
import * as tf from '@tensorflow/tfjs'

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

function App() {
  const [modelState, setModelState] = useState<'none'|'loading'|'ready'|'inferencing'>('none');
  const pipeline = useRef<StableDiffusionPipeline|null>(null)
  useEffect(() => {
    if (typeof window !== 'undefined') {
      // debouncedInference()
    }
  }, [])

  const progressCallback = async (info: ProgressCallback) => {
    if (info.images) {
      const canvas = document.getElementById('canvas')
      if (canvas) {
        await tf.browser.toPixels(info.images[0], canvas as HTMLCanvasElement)
      }
    }
  }
  const loadModel = () => {
    if (!window.confirm('This will download approximately 3.5gb and use 8gb of your RAM. Are you sure want to continue?')) {
      return
    }
    setModelState('loading')
    StableDiffusionPipeline.fromPretrained('wasm', '/aislamov/stable-diffusion-2-1-base-onnx')
      .then((p) => {
        pipeline.current = p
        setModelState('ready')
        p.run("a horse", "", 1, 7.5, 3, true, progressCallback)
      })
  }
  return (
    <ThemeProvider theme={darkTheme}>
      <CssBaseline enableColorScheme={true} />
      <Container>
        <Alert severity="error">Please make sure you are running canary chrome with enabled "Unsafe WebGPU Support", "Experimental WebAssembly", and "Experimental WebAssembly JavaScript Promise Integration (JSPI)" flags!</Alert>
        <Box sx={{ bgcolor: '#282c34', height: '100vh' }}>
          <Grid container spacing={2}>
            <Grid item xs={6}>
              <Stack>
                <TextField label="Prompt" variant="standard" />
                <TextField label="Negative Prompt" variant="standard" />
              </Stack>
            </Grid>
            <Grid item xs={6}>
              <canvas id={'canvas'} width={128} height={128} />
            </Grid>
          </Grid>
          <Button variant="outlined" onClick={loadModel}>Load model</Button>
        </Box>
      </Container>
    </ThemeProvider>
  );
}

export default App;
