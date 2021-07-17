import pygame

while True:
	pygame.mixer.init()
	pygame.mixer.set_num_channels(8)
	voice = pygame.mixer.Channel(5)

	sound = pygame.mixer.Sound("warn.wav")
	
	
	
	if voice.get_busy() == 0:
		voice.play(sound)