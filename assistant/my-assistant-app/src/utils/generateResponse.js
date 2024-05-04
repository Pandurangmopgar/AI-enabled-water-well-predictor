import { GoogleGenerativeAI } from '@google/generative-ai';

// Access your API key as an environment variable (see "Set up your API key" above)
const genAI = new GoogleGenerativeAI('AIzaSyCvIKmj2RJkIyfZsu7mu7TLzsJwdmBr19Y');

export const generateResponseFromAPI = async (input) => {
  const model = genAI.getGenerativeModel({ model: 'gemini-pro' });
  const result = await model.generateContent(input);
  return await result.response.text();
};