import React, { useState } from 'react';
import { GoogleGenerativeAI } from '@google/generative-ai';
import Header from './Header';
import ConversationArea from './ConversationArea';
import InputArea from './InputArea';
import styled from 'styled-components';

// Access your API key as an environment variable (see "Set up your API key" above)
const genAI = new GoogleGenerativeAI('AIzaSyCvIKmj2RJkIyfZsu7mu7TLzsJwdmBr19Y');
// console.log(profile)
const ChatAppContainer = styled.div`
  display: flex;
  flex-direction: column;
  height: 100vh;
  background-color: #f5f5f5;
  font-family: 'Roboto', sans-serif;
`;

const ChatApp = ({ profile }) => {
  console.log(profile);
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
      const userMessage = {
        type: 'user',
        text: prompt,
        name: profile?.name || 'User', // Use the user's name from the profile, or fallback to 'User'
        avatar: profile?.imageUrl || 'default-avatar.png', // Use the user's avatar from the profile, or fallback to a default avatar
      };
      const assistantMessage = {
        type: 'assistant',
        text,
        name: 'Assistant', // You can customize the assistant's name here
        avatar: 'assistant-avatar.png', // You can set the assistant's avatar image here
      };
      setMessages([...messages, userMessage, assistantMessage]);
      setPrompt('');
    }
  };

  return (
    <ChatAppContainer>
      <Header profile={profile} />
      <ConversationArea messages={messages} />
      <InputArea prompt={prompt} onPromptChange={handlePromptChange} onSubmit={handleSubmit} />
    </ChatAppContainer>
  );
};

export default ChatApp;