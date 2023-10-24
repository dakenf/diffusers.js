import Box from '@mui/material/Box'
import List from '@mui/material/List'
import ListItem from '@mui/material/ListItem'
import React from 'react'
import ListItemText from '@mui/material/ListItemText'
import Divider from '@mui/material/Divider'


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

export function FAQ () {
  return (
    <Box sx={{ width: '100%', bgcolor: 'background.paper' }}>
      <List>
        <ListItem>
          <h2>FAQ</h2>
        </ListItem>
        <FaqItem question={'What if I get protobuf parsing failed error?'} answer={'Open DevTools, go to Application -> Storage and press "Clear site data".'} />
        <FaqItem question={'What if I get sbox_fatal_memory_exceeded?'} answer={"You don't have enough RAM to run SD. You can try reloading the tab or browser."} />
        <FaqItem question={'How did you make it possible?'} answer={'In order to run it, I had to port StableDiffusionPipeline from python to JS. Then patch onnxruntime and emscripten+binaryen (WebAssembly compiler toolchain) to support allocating and using >4GB memory. Once my pull requests get to release, anyone would be able to compile and run code that uses >4GB in the browser.'} />
        <FaqItem question={'Can I run it locally?'} answer={'Yes, this page’s code is available here: https://github.com/dakenf/stable-diffusion-webgpu-minimal'} />
        <FaqItem question={'Can I use your patched onnxruntime to run big LLMs with transformers.js?'} answer={'Yes, you can use this package but i don’t guarantee it will be working in all cases. This build is limited to 8GB of memory, so you can load up to ~4GB weights. Just use https://www.npmjs.com/package/@aislamov/onnxruntime-web64'} />
        <FaqItem question={'Are you going to make a pull request in onnxruntime repo?'} answer={'Yes. It will be my second one, i’ve added GPU acceleration to node.js binding earlier.'} />
      </List>
    </Box>
  )
}
