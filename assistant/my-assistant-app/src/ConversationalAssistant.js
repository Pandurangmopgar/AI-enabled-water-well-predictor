import React, { useState } from 'react';
import { GPT2LMHeadModel, GPT2Tokenizer } from 'transformers';

const ConversationalAssistant = () => {
  const [model, setModel] = useState(GPT2LMHeadModel.from_pretrained('gpt2'));
  const [tokenizer, setTokenizer] = useState(GPT2Tokenizer.from_pretrained('gpt2'));
  const [userInput, setUserInput] = useState('');
  const [assistantResponse, setAssistantResponse] = useState('');

  const generateResponse = async () => {
    const input_ids = tokenizer.encode(userInput, return_tensors='pt');
    const output = await model.generate(input_ids, max_length=150, num_return_sequences=1, do_sample=true, top_k=50, top_p=0.95, num_return_sequences=1);
    const response = tokenizer.decode(output[0], skip_special_tokens=true);
    setAssistantResponse(response);
  };

  return (
    <div>
      <h1>Conversational Assistant</h1>
      <input
        type="text"
        value={userInput}
        onChange={(e) => setUserInput(e.target.value)}
      />
      <button onClick={generateResponse}>Send</button>
      <div>{assistantResponse}</div>
    </div>
  );
};

export default ConversationalAssistant;