import gymnasium as gym
import time
import ale_py
import keyboard
import os

# Configuración
ENV_NAME = "ALE/Skiing-v5"
gym.register_envs(ale_py)
env = gym.make(ENV_NAME, render_mode="human")
obs, info = env.reset()

# Empezamos mirando la dirección 107 (Sospechoso habitual en Atari)
# Si no es esta, usa las flechas ARRIBA/ABAJO para cambiar.
watched_index = 107 
last_val = 0

print("\n" + "="*60)
print("TV RAM: VIGILANCIA DE UNA SOLA VARIABLE")
print("CONTROLES:")
print("   [<] IZQUIERDA / [>] DERECHA : Mover Esquiador")
print("   [^] ARRIBA    / [v] ABAJO   : Cambiar canal de memoria (+/- 1)")
print("="*60 + "\n")

running = True

while running:
    # 1. Controles de Juego y TV
    action = 0
    
    # Movimiento del Esquiador
    if keyboard.is_pressed('left'): action = 1
    elif keyboard.is_pressed('right'): action = 2
    elif keyboard.is_pressed('esc'): running = False
    
    # Cambiar de canal (Variable RAM a vigilar)
    # Ponemos un pequeño sleep para que no salte 50 números de golpe
    if keyboard.is_pressed('up'): 
        watched_index = (watched_index + 1) % 128
        print(f"\n--- CAMBIANDO A RAM[{watched_index}] ---")
        time.sleep(0.2) 
        
    elif keyboard.is_pressed('down'): 
        watched_index = (watched_index - 1) % 128
        print(f"\n--- CAMBIANDO A RAM[{watched_index}] ---")
        time.sleep(0.2)

    # 2. Paso del juego
    obs, reward, terminated, truncated, info = env.step(action)
    
    # 3. Lectura de RAM
    ram = env.unwrapped.ale.getRAM()
    current_val = ram[watched_index]

    # 4. Mostrar en pantalla (Solo si cambia o para refrescar)
    # Usamos colores simples si tu consola lo soporta, o asteriscos
    if current_val != last_val:
        # ¡CAMBIO DETECTADO!
        print(f"RAM[{watched_index:03d}] CAMBIO: {last_val} -> {current_val}   (¿Fue al pasar bandera?)")
    else:
        # Muestra constante para que sepas qué miras (con \r para no llenar la pantalla)
        print(f"\rVigilando RAM[{watched_index:03d}] Valor Actual: {current_val}      ", end="")
    
    last_val = current_val

    # Velocidad jugable
    time.sleep(0.04)

    if terminated or truncated:
        env.reset()
        print("\n--- Reiniciando ---")

env.close()