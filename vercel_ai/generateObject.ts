import { generateObject } from 'ai';
import { z } from 'zod';
import { createOpenAICompatible } from "@ai-sdk/openai-compatible";


const localLLM = createOpenAICompatible({
  name: 'localLLM',
  baseURL: 'http://127.0.0.1:8080',
})

const { object } = await generateObject({
  model: localLLM('localLLM'),
  schema: z.object({
    names: z.array(z.string()).length(3).describe('A list of three names'),
  }),
  prompt: 'Generate three names',
});

console.log(JSON.stringify(object, null, 2));
