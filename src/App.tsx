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
        <FaqItem question={'Can I run it locally?'} answer={'Yes, this page’s code is available here: '} />
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
  const [status, setStatus] = useState('Ready');
  const pipeline = useRef<StableDiffusionPipeline|null>(null)
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
    StableDiffusionPipeline.fromPretrained('webgpu', '/aislamov/stable-diffusion-2-1-base-onnx', progressCallback)
      .then((p) => {
        pipeline.current = p
        setModelState('ready')
      })
      .catch(e => alert(e.message))
  }

  const runInference = () => {
    if (!pipeline.current) {
      return
    }
    setModelState('inferencing')

    pipeline.current.run({
      prompt: prompt,
      negativePrompt: negativePrompt,
      numInferenceSteps: inferenceSteps,
      width: 512,
      height: 512,
      runVaeOnEachStep: true,
      progressCallback,
    }).then(images => {
      progressCallback({
        step: 'Done',
      //   images,
      })
      setModelState('ready')
    })
  }
  return (
    <ThemeProvider theme={darkTheme}>
      <CssBaseline enableColorScheme={true} />
      <Container>
        <Stack>
          <Alert severity="error">You need Chrome Canary with "Unsafe WebGPU Support", "Experimental WebAssembly", and "Experimental WebAssembly JavaScript Promise Integration (JSPI)" flags!</Alert>
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
                  disabled={modelState != 'ready'}
                  onChange={(e) => setInferenceSteps(parseInt(e.target.value))}
                  value={inferenceSteps}
                />
                <p>Each step takes about 1 minute + ~10sec to run VAE decoder to generate image. Having DevTools open will slow everything down to about 2x.
                  UNET runs only on CPU (it's 10% faster and does not give correct results on GPU), so will hang the browser tab.</p>
                <p>Minimum number of steps to get an acceptable result is 20. However, 3 would be fine for demo purposes.</p>
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
