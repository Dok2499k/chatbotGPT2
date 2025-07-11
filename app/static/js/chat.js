const chatEl = document.getElementById('chat');
const inputEl = document.getElementById('user-input');
const btnEl = document.getElementById('send-btn');

function appendMessage(who, text) {
  const div = document.createElement('div');
  div.className = who === 'You' ? 'text-end mb-2' : 'text-start mb-2';
  div.innerHTML = `<strong>${who}:</strong> ${text}`;
  chatEl.appendChild(div);
  chatEl.scrollTop = chatEl.scrollHeight;
}

btnEl.addEventListener('click', async () => {
  const prompt = inputEl.value.trim();
  if (!prompt) return;
  appendMessage('You', prompt);
  inputEl.value = '';

  try {
    const res = await fetch('/predict', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({user_input: prompt})
    });
    const data = await res.json();
    if (data.error) {
      appendMessage('Error', data.error);
    } else {
      appendMessage('Bot', data.response);
    }
  } catch (e) {
    appendMessage('Error', 'Сервер недоступен');
  }
});

inputEl.addEventListener('keydown', e => {
  if (e.key === 'Enter') btnEl.click();
});