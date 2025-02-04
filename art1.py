import pygame
import random

# Initialisation de Pygame
pygame.init()
width, height = 800, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Generative Art - Shapes")

# Couleurs possibles
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]

# Fonction pour dessiner un triangle
def draw_triangle(surface, color, x, y, size):
    points = [(x, y), (x + size, y + size), (x - size, y + size)]
    pygame.draw.polygon(surface, color, points)

# Génération aléatoire de formes
for _ in range(100):  # Dessiner 100 formes
    shape_type = random.choice(["circle", "square", "rectangle", "triangle"])
    x = random.randint(50, width - 50)
    y = random.randint(50, height - 50)
    color = random.choice(colors)

    if shape_type == "circle":
        radius = random.randint(10, 50)
        pygame.draw.circle(screen, color, (x, y), radius)

    elif shape_type == "square":
        size = random.randint(20, 50)
        pygame.draw.rect(screen, color, (x, y, size, size))

    elif shape_type == "rectangle":
        w, h = random.randint(30, 80), random.randint(20, 60)
        pygame.draw.rect(screen, color, (x, y, w, h))

    elif shape_type == "triangle":
        size = random.randint(20, 50)
        draw_triangle(screen, color, x, y, size)

# Sauvegarde de l'image
pygame.image.save(screen, "static/art_shapes.png")
pygame.quit()
