import React from 'react';
import styled from 'styled-components';

const SpeechButton = ({ onClick, children }) => {
  return <SpeechButtonContainer onClick={onClick}>{children}</SpeechButtonContainer>;
};

export default SpeechButton;

const SpeechButtonContainer = styled.div`
  position: fixed;
  bottom: 1rem;
  right: 1rem;
  width: 3rem;
  height: 3rem;
  border-radius: 50%;
  background-color: #007bff;
  color: #fff;
  display: flex;
  justify-content: center;
  align-items: center;
  font-size: 1.5rem;
  cursor: pointer;
  box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
`;