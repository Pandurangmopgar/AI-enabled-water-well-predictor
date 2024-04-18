import React from 'react';
import styled from 'styled-components';

const InputArea = ({ prompt, onPromptChange, onSubmit }) => {
  return (
    <InputContainer onSubmit={onSubmit}>
      <Input
        type="text"
        placeholder="Type your message..."
        value={prompt}
        onChange={onPromptChange}
      />
      <SubmitButton type="submit">Send</SubmitButton>
    </InputContainer>
  );
};

export default InputArea;

const InputContainer = styled.form`
  display: flex;
  align-items: center;
  padding: 1rem;
  background-color: #fff;
  box-shadow: 0 -2px 5px rgba(0, 0, 0, 0.1);
`;

const Input = styled.input`
  flex-grow: 1;
  padding: 0.5rem 1rem;
  border: none;
  border-radius: 1rem;
  background-color: #f5f5f5;
  font-size: 1rem;
  &:focus {
    outline: none;
  }
`;

const SubmitButton = styled.button`
  margin-left: 0.5rem;
  padding: 0.5rem 1rem;
  border: none;
  border-radius: 1rem;
  background-color: #007bff;
  color: #fff;
  font-size: 1rem;
  cursor: pointer;
  &:hover {
    background-color: #0056b3;
  }
`;