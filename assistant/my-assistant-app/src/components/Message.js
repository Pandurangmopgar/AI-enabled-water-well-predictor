import React from 'react';
import styled from 'styled-components';

const Message = ({ type, text }) => {
  return (
    <MessageContainer type={type}>
      <MessageText>{text}</MessageText>
    </MessageContainer>
  );
};

export default Message;

const MessageContainer = styled.div`
  display: flex;
  justify-content: ${(props) => (props.type === 'user' ? 'flex-end' : 'flex-start')};
  margin-bottom: 0.5rem;
`;

const MessageText = styled.div`
  max-width: 70%;
  padding: 0.5rem 1rem;
  border-radius: 1rem;
  background-color: ${(props) => (props.type === 'user' ? '#007bff' : '#fff')};
  color: ${(props) => (props.type === 'user' ? '#fff' : '#333')};
  box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
`;