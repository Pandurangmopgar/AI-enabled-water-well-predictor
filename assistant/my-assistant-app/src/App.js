import React from 'react';
import {
  ChatAppContainer,
  Header,
  Title,
  UserAvatar,
  ConversationContainer,
  MessageContainer,
  MessageText,
  InputContainer,
  Input,
  SubmitButton,
} from './styles/ChatApp.styled';
import ChatApp from './components/ChatApp';

const App = () => {
  return (
    <ChatAppContainer>
      <ChatApp />
    </ChatAppContainer>
  );
};

export default App;