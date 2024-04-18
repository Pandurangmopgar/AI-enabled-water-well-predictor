import React from 'react';
import styled from 'styled-components';
import Message from './Message';

const ConversationArea = ({ messages }) => {
  return (
    <ConversationContainer>
      {messages.map((message, index) => (
        <Message key={index} type={message.type} text={message.text} />
      ))}
    </ConversationContainer>
  );
};

export default ConversationArea;

const ConversationContainer = styled.div`
  flex-grow: 1;
  padding: 1rem;
  overflow-y: auto;
`;