import React, { useEffect } from 'react'
import logo from './logo.svg';
import { Tensor, InferenceSession } from "onnxruntime-web";
import './App.css';
import { StableDiffusionPipeline } from './lib/StableDiffusionPipeline'
import { debounce } from 'lodash'

async function inference () {
  try {
    const pipe = await StableDiffusionPipeline.fromPretrained('webnn', '/aislamov/stable-diffusion-2-1-base-onnx');
    const result = pipe.run("a horse", "", 1, 7.5, 30)
    console.log(result);
  } catch (e) {
    console.error(e)
  }
}

const debouncedInference = debounce(inference, 1000)

function App() {
  useEffect(() => {
    if (typeof window !== 'undefined') {
      debouncedInference()
    }
  }, [])
  return (
    <div className="App">
      <header className="App-header">
        <img src={logo} className="App-logo" alt="logo" />
        <p>
          Edit <code>src/App.tsx</code> and save to reload.
        </p>
        <a
          className="App-link"
          href="https://reactjs.org"
          target="_blank"
          rel="noopener noreferrer"
        >
          Learn React
        </a>
      </header>
    </div>
  );
}

export default App;
