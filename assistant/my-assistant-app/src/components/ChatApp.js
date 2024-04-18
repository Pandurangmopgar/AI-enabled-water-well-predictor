import React, { useState } from 'react';
import { GoogleGenerativeAI } from '@google/generative-ai';
import Header from './Header';
import ConversationArea from './ConversationArea';
import InputArea from './InputArea';
import styled from 'styled-components';

// Access your API key as an environment variable (see "Set up your API key" above)
const genAI = new GoogleGenerativeAI('AIzaSyCvIKmj2RJkIyfZsu7mu7TLzsJwdmBr19Y');

const ChatAppContainer = styled.div`
  display: flex;
  flex-direction: column;
  height: 100vh;
  background-color: #f5f5f5;
  font-family: 'Roboto', sans-serif;
`;

const ChatApp = () => {
  const [prompt, setPrompt] = useState('');
  const [messages, setMessages] = useState([]);

  const handlePromptChange = (e) => {
    setPrompt(e.target.value);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (prompt.trim() !== '') {
      const model = genAI.getGenerativeModel({ model: 'gemini-pro' });
      const result = await model.generateContent(prompt);
      const text = await result.response.text();
      setMessages([...messages, { type: 'user', text: prompt }, { type: 'assistant', text }]);
      setPrompt('');
    }
  };

  return (
    <ChatAppContainer>
      <Header />
      <ConversationArea messages={messages} />
      <InputArea prompt={prompt} onPromptChange={handlePromptChange} onSubmit={handleSubmit} />
    </ChatAppContainer>
  );
};

export default ChatApp;