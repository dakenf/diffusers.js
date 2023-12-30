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
import { Checkbox, FormControl, InputLabel, MenuItem, Select } from '@mui/material'
import { FormControlLabel } from '@mui/material';
import { BrowserFeatures, hasFp16 } from './components/BrowserFeatures'
import { FAQ } from './components/FAQ'
import { Tensor } from '@xenova/transformers'
import cv from '@techstark/opencv-js'
import { StableDiffusionControlNetPipeline } from '../../../dist/pipelines/StableDiffusionControlNetPipeline';
import { getBlobFromImage, generateColors, segArgmax, posePostProcess, loadAnnotatorFile } from './annotator_helper_functions' 

const darkTheme = createTheme({
  palette: {
    mode: 'dark',
  },
});

interface SelectedPipeline {
  name: string
  repo: string
  revision: string
  fp16: boolean
  steps: number
  hasImg2Img: boolean
  hasControlNet: boolean
  controlnet?: string
}

const pipelines = [
  {
    name: 'LCM Dreamshaper FP16 (2.2GB)',
    repo: 'aislamov/lcm-dreamshaper-v7-onnx',
    revision: 'main',
    fp16: true,
    width: 768,
    height: 768,
    steps: 8,
    hasImg2Img: false,
    hasControlNet: false
  },
  // {
  //   name: 'LCM Dreamshaper FP32 (4.2GB)',
  //   repo: 'aislamov/lcm-dreamshaper-v7-onnx',
  //   revision: 'fp32',
  //   fp16: false,
  //   width: 768,
  //   height: 768,
  //   steps: 8,
  // },
  {
    name: 'StableDiffusion 2.1 Base FP16 (2.6GB)',
    repo: 'aislamov/stable-diffusion-2-1-base-onnx',
    revision: 'main',
    fp16: true,
    width: 512,
    height: 512,
    steps: 20,
    hasImg2Img: true,
    hasControlNet: false
  },
  // {
  //   name: 'StableDiffusion 2.1 Base FP32 (5.1GB)',
  //   repo: 'aislamov/stable-diffusion-2-1-base-onnx',
  //   revision: 'fp32',
  //   fp16: false,
  //   width: 512,
  //   height: 512,
  //   steps: 20,
  // },
  {
    name: 'StableDiffusion 1.5 Base FP16 Canny (2.9GB)',
    repo: 'jdp8/stable-diffusion-1-5-canny-base-onnx',
    revision: 'main',
    fp16: true,
    width: 512,
    height: 512,
    steps: 20,
    hasImg2Img: true,
    hasControlNet: true,
    controlnet: 'canny'
  },
  {
    name: 'StableDiffusion 1.5 Base FP16 Semantic Segmentation (2.9GB)',
    repo: 'jdp8/stable-diffusion-1-5-seg-v11p-onnx',
    revision: 'main',
    fp16: true,
    width: 512,
    height: 512,
    steps: 20,
    hasImg2Img: true,
    hasControlNet: true,
    controlnet: 'semantic_segmentation'
  },
  {
    name: 'StableDiffusion 1.5 Base FP16 OpenPose (2.9GB)',
    repo: 'jdp8/stable-diffusion-1-5-openpose-v11p-onnx',
    revision: 'main',
    fp16: true,
    width: 512,
    height: 512,
    steps: 20,
    hasImg2Img: true,
    hasControlNet: true,
    controlnet: 'openpose'
  },
]

function App() {
  const [hasF16, setHasF16] = useState<boolean>(false);
  const [selectedPipeline, setSelectedPipeline] = useState<SelectedPipeline|undefined>(pipelines[0]);
  const [modelState, setModelState] = useState<'none'|'loading'|'ready'|'inferencing'>('none');
  const [prompt, setPrompt] = useState('An astronaut riding a horse');
  const [negativePrompt, setNegativePrompt] = useState('');
  const [inferenceSteps, setInferenceSteps] = useState(20);
  const [guidanceScale, setGuidanceScale] = useState(7.5);
  const [seed, setSeed] = useState('');
  const [status, setStatus] = useState('Ready');
  const pipeline = useRef<StableDiffusionXLPipeline|StableDiffusionPipeline|StableDiffusionControlNetPipeline|null>(null);
  const [img2img, setImg2Img] = useState(false);
  const [inputImage, setInputImage] = useState<Float32Array>();
  const [strength, setStrength] = useState(0.8);
  const [controlNetImage, setControlNetImage] = useState<Float32Array>();
  const [annotator_model, setAnnotatorModel] = useState('');
  const [annotator_config, setAnnotatorConfig] = useState('');
  const [runVaeOnEachStep, setRunVaeOnEachStep] = useState(false);
  useEffect(() => {
    setModelCacheDir('models')
    hasFp16().then(v => {
      setHasF16(v)
      if (v === false) {
        setSelectedPipeline(pipelines.find(p => p.fp16 === false))
      }
    })
  }, [])

  useEffect(() => {
    setInferenceSteps(selectedPipeline?.steps || 20)
  }, [selectedPipeline])

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
      // @ts-ignore
      await drawImage(info.images[0])
    }
  }

  const loadModel = async () => {
    if (!selectedPipeline) {
      return
    }
    setModelState('loading')
    try {
      if (pipeline.current) {
        // @ts-ignore
        pipeline.current.release()
      }
      pipeline.current = await DiffusionPipeline.fromPretrained(
        selectedPipeline.repo,
        {
          revision: selectedPipeline?.revision,
          progressCallback
        }
      )
      setModelState('ready')
    } catch (e) {
      alert(e)
      console.error(e)
    }
  }

  /**
   * Extracts the RGB data from an RGBA image array.
   * 
   * @param d RGBA image array.
   * @param normalize Normalizes the image array to [-1,1] if true. Set to true for img2img and false for controlnet.
   * @returns RGB Float32Array.
   */
  function getRgbData(d: Uint8ClampedArray, normalize=true) {
    let rgbData: any = [[], [], []]; // [r, g, b]
    // remove alpha and put into correct shape:
    for(let i = 0; i < d.length; i += 4) {
        let x = (i/4) % 512;
        let y = Math.floor((i/4) / 512)
        if(!rgbData[0][y]) rgbData[0][y] = [];
        if(!rgbData[1][y]) rgbData[1][y] = [];
        if(!rgbData[2][y]) rgbData[2][y] = [];
        rgbData[0][y][x] = normalize ? (d[i+0]/255) * 2 - 1 : (d[i+0]/255);
        rgbData[1][y][x] = normalize ? (d[i+1]/255) * 2 - 1 : (d[i+1]/255);
        rgbData[2][y][x] = normalize ? (d[i+2]/255) * 2 - 1 : (d[i+2]/255);
    }
    rgbData = Float32Array.from(rgbData.flat().flat());
    return rgbData;
  }

  /**
   * Takes an input image and saves it to the corresponding state variable. 
   * The input image can be used either for the controlnet or img2img pipelines
   * which is determined by type.
   * 
   * @param e HTML file upload element.
   * @param type Pipeline of the input image.
   * @returns void
   */
  function uploadImage(e: any, type: 'controlnet'|'img2img', controlnet='canny') {
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
        if (type == 'img2img') {
          const imageCanvas = document.createElement('canvas');
          imageCanvas.width = uploadedImage.width;
          imageCanvas.height = uploadedImage.height;
          const imgCtx = imageCanvas.getContext('2d') as CanvasRenderingContext2D;
          imgCtx.drawImage(uploadedImage, 0, 0, uploadedImage.width, uploadedImage.height);
          const imageData = imgCtx.getImageData(0, 0, uploadedImage.width, uploadedImage.height).data;

          const rgb_array = getRgbData(imageData);
          setInputImage(rgb_array);
        }
        else if(type == 'controlnet') {
          const cvImg = cv.imread(uploadedImage); // RGBA Image | 4 Channels

          if(controlnet == 'semantic_segmentation') {
            const inputSize = [513, 513];
            const mean = [127.5, 127.5, 127.5];
            const std = 0.007843;
            const swapRB = false;

            const input = getBlobFromImage(inputSize, mean, std, swapRB, cvImg);
            const net = cv.readNet(annotator_model);
            net.setInput(input);
            const result = net.forward();
            const colors = generateColors(result);
            const output = segArgmax(result, colors);
            const resizedOutput = new cv.Mat();
            const dsize = new cv.Size(512, 512);
            cv.resize(output, resizedOutput, dsize, 0, 0, cv.INTER_AREA);
            const rgbSem = getRgbData(Uint8ClampedArray.from(resizedOutput.data), false);
            setControlNetImage(rgbSem);
            cvImg.delete();input.delete();net.delete();result.delete();resizedOutput.delete();
          }
          else if(controlnet == 'openpose') {
            // inputSize can be changed, the original is [368, 368]. The higher the size, the slower the annotation and vice-versa.
            const inputSize = [125, 125];
            const mean = [0, 0, 0];
            const std = 0.00392;
            const swapRB = false;
            const threshold = 0.1;

            // the pairs of keypoint, can be "COCO", "MPI" and "BODY_25"
            let dataset = '';

            if(annotator_model == 'pose_iter_584000.caffemodel') {
              dataset = 'BODY_25'
            }
            else if(annotator_model == 'pose_iter_440000.caffemodel') {
              dataset = 'COCO'
            }
            else if (annotator_model == 'pose_iter_160000.caffemodel') {
              dataset = 'MPI'
            }

            const input = getBlobFromImage(inputSize, mean, std, swapRB, cvImg);
            let net = cv.readNet(annotator_model, annotator_config);
            net.setInput(input);
            const result = net.forward();
            const output = posePostProcess(result, dataset, threshold, 512, 512);
            const rgbaPose = new cv.Mat();
            cv.cvtColor(output, rgbaPose, cv.COLOR_RGB2RGBA, 0); // RGBA Pose Image | 4 Channels
            const rgbPose = getRgbData(Uint8ClampedArray.from(rgbaPose.data), false);
            setControlNetImage(rgbPose);
            cvImg.delete();input.delete();net.delete();result.delete();rgbaPose.delete();
          }
          else if(controlnet == 'canny') {
            const imgGray = new cv.Mat();
            cv.cvtColor(cvImg, imgGray, cv.COLOR_RGBA2GRAY); // Gray Image | 1 Channel
            const imgCanny = new cv.Mat();
            cv.Canny(imgGray, imgCanny, 100, 200, 3, false); // Canny Image | 1 Channel
            const rgbaCanny = new cv.Mat();
            cv.cvtColor(imgCanny, rgbaCanny, cv.COLOR_GRAY2RGBA, 0); // RGBA Canny Image | 4 Channels
            const rgbCanny = getRgbData(Uint8ClampedArray.from(rgbaCanny.data), false);
            setControlNetImage(rgbCanny);
            cvImg.delete();imgGray.delete();imgCanny.delete();rgbaCanny.delete();
          }
        }
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
      strength: strength,
      controlNetImage: controlNetImage
    })
    await drawImage(images[0])
    setModelState('ready')
  }

  return (
    <ThemeProvider theme={darkTheme}>
      <CssBaseline enableColorScheme={true} />
      <Container>
        <BrowserFeatures />
        <Stack alignItems={'center'}>
          <p>Built with <a href={"https://github.com/dakenf/diffusers.js"} target={"_blank"}>diffusers.js</a></p>
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
                {selectedPipeline?.hasControlNet &&
                  (
                    <>
                      <label htmlFor="upload_annotator_model">Upload Annotator Model for ControlNet Pipeline:</label>
                      <TextField
                        id="upload_annotator_model"
                        inputProps={{accept:".caffemodel,.pb"}}
                        type={"file"}
                        disabled={modelState != 'ready'}
                        onChange={async (e) => {
                          const fileName = await loadAnnotatorFile(e)
                          //@ts-ignore
                          setAnnotatorModel(fileName)
                        }}
                      />

                      {selectedPipeline?.controlnet == 'openpose' && 
                        (
                          <>
                            <label htmlFor="upload_config">Upload Annotator Config File for ControlNet Pipeline:</label>
                            <TextField
                              id="upload_config"
                              inputProps={{accept:".prototxt"}}
                              type={"file"}
                              disabled={modelState != 'ready'}
                              onChange={async (e) => {
                                const fileName = await loadAnnotatorFile(e)
                                //@ts-ignore
                                setAnnotatorConfig(fileName)
                              }}
                            />
                          </>
                      )}
                      <label htmlFor="upload_controlnet_image">Upload Image for ControlNet Pipeline (annotator files must be uploaded first):</label>
                      <TextField
                        id="upload_controlnet_image"
                        inputProps={{accept:"image/*"}}
                        type={"file"}
                        disabled={modelState != 'ready'}
                        onChange={(e) => uploadImage(e, "controlnet", selectedPipeline.controlnet)}
                      />
                  </>
                )}
                {selectedPipeline?.hasImg2Img &&
                  (
                    <>
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
                        onChange={(e) => uploadImage(e, 'img2img')}
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
                    </>
                )}
                <FormControlLabel
                  label="Check if you want to run VAE after each step"
                  control={<Checkbox
                    disabled={modelState != 'ready'}
                    onChange={(e) => setRunVaeOnEachStep(e.target.checked)}
                    checked={runVaeOnEachStep}
                  />}
                />
                <FormControl fullWidth>
                  <InputLabel id="demo-simple-select-label">Pipeline</InputLabel>
                    <Select
                      value={selectedPipeline?.name}
                      onChange={e => {
                        setSelectedPipeline(pipelines.find(p => e.target.value === p.name))
                        setModelState('none')
                      }}>
                      {pipelines.map(p => <MenuItem value={p.name} disabled={!hasF16 && p.fp16}>{p.name}</MenuItem>)}
                    </Select>
                </FormControl>
                <p>Press the button below to download model. It will be stored in your browser cache.</p>
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
