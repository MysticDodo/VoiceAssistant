# Basic Voice Assistant #
## Tech Stack ##
- Custom trained Hugging Face Transformer model for getting the intent of voice command.
- SpaCy model for getting the entity of the command.
- Speech Recognition for converted speech to text, uses the google option.

## To set up: ##
- Run the model.py to train the mode (needs to be done only once).
- Then run the main.
- Uses an inefficient and simple wake word (set to just "hey" because Speech Recognition failed to understand the name Gonkfield - not surprising).
- Then speak and wait for a response.

## Commands: ##
- Time - tells the time.
- Play music - tells you it's playing music, if you give a genre it recognises the genre, but doesn't actually play any music.
- Open app - might recognise other apps, but will only open spotify or youtube, and it opens the website not the app.
- Set a timer - tells you it has set a timer for x amount of time you give it, if you don't give it a time it will ask how long, but context is not kept so it does nothing with it.
- More commands to be added and model to be improved.
