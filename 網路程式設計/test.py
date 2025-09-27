import pygame
import sys
import os
import time
import threading
import socket
pygame.init()

### 初始化 a=列, b=行, l=格子大小, R=球半徑  ###           
a = b = 10
l = 40
R = 15
vel = 40
now = [1, 1] #x,y
test_Blue_open=1
test_Red_open=1
WIDTH = l*a+240
HEIGHT = l*b+80
gamestate = 0  # 遊戲狀態 0=遊戲前, 1=遊戲中, 2=遊戲結束
font_name = os.path.join("font.ttf")

#按鈕
font = pygame.font.Font(font_name, l-10)
text1 = font.render(" POP ", True, 'black')
rect1 = text1.get_rect(topleft=(WIDTH-115,HEIGHT/2-l*2))

text2 = font.render(" Respawn ", True, 'black')
rect2 = text2.get_rect(topleft=(WIDTH-148,HEIGHT/2+(l*2)))

text3 = font.render(" cheat ", True, 'black')
rect3 = text3.get_rect(topleft=(WIDTH-125,HEIGHT/2+(l*4)))

### 設置顏色 ###
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
color_text=(255, 255, 255)

white = pygame.Surface((l, l), flags=pygame.HWSURFACE); black = pygame.Surface((l, l), flags=pygame.HWSURFACE); blue = pygame.Surface((l, l), flags=pygame.HWSURFACE)
block = pygame.Surface((l*5, l), flags=pygame.HWSURFACE); green = pygame.Surface((l, l), flags=pygame.HWSURFACE); red = pygame.Surface((l, l), flags=pygame.HWSURFACE)
white.fill(color='white'); black.fill(color='black');blue.fill(color='blue')
block.fill(color='white'); green.fill(color='green'); red.fill(color='red')

def renew():
    screen.blit(text1, rect1)#按鈕
    pygame.draw.rect(screen, (0,0,255),rect1,2)#框
    screen.blit(text2, rect2)#按鈕
    pygame.draw.rect(screen, (0,0,255),rect2,2)#框
    screen.blit(text3, rect3)#按鈕
    pygame.draw.rect(screen, (0,0,255),rect3,2)#框
    ## 黑暗模式 ##
    for i in range(a+2):
        for j in range(b+2):
            screen.blit(black, (i*l, j*l))
    screen.blit(white, (now[0]*l, now[1]*l))
    pygame.display.flip()
    # 顏色 0=white 1=black 2=red 3=green 4=player 5=blue 6=gray
    for i in range(now[0]-1,now[0]+2):
        for j in range(now[1]-1,now[1]+2):
            if(i>a or j>b):
                pass
            elif be[i][j]==0:
                screen.blit(white, (i*l, j*l))
            elif be[i][j]==1:
                screen.blit(black, (i*l, j*l))
            elif be[i][j]==2:
                screen.blit(red, (i*l, j*l))
            elif be[i][j]==3:
                screen.blit(green, (i*l, j*l))
            elif be[i][j]==4:
                pygame.draw.circle(screen, (255,255,55), (now[0]*l+l/2,now[1]*l+l/2), R)
            elif be[i][j]==5:
                screen.blit(blue, (i*l, j*l))
    pygame.display.flip()

def win():
    print("!!! YOU WIN !!!")
    be[now[0]][now[1]]=0
    be[1][1]=4
    now[0]=1; now[1]=1
    renew()

def restart():
    be[now[0]][now[1]]=0
    be[1][1]=4
    now[0]=1; now[1]=1
    renew()
    if player == "blue": txt_Player="blue"; color=(0,0,255)
    else: txt_Player="red"; color=(255,0,0)
    draw_text(screen, txt_Player, l, color, WIDTH-77,HEIGHT/2-(l*4))
    pygame.display.flip()

def die():
    die_txt=font.render(" YOU DIE !!! ", True, 'black')
    rect_die = die_txt.get_rect(topleft=(WIDTH-165,HEIGHT/2+l))
    screen.blit(die_txt, rect_die)
    be[now[0]][now[1]]=0
    be[1][1]=4
    now[0]=1; now[1]=1
    renew(); time.sleep(0.3)
    screen.blit(block, rect_die)

def control_Prick(list,n,test): # list = 要改變的列表, n = 顏色, test = 是否偵測玩家
    global red_prick
    global blue_prick
    for i in range (len(list)):
        if test == 1 and be[list[i][0]][list[i][1]] == 4:
            die()
        if be[list[i][0]][list[i][1]] == n:
            be[list[i][0]][list[i][1]] = 0
        elif be[list[i][0]][list[i][1]] == 0:
            be[list[i][0]][list[i][1]] = n

def draw_text(surf, text, size, color, x, y):  # 在畫面上寫字
    font = pygame.font.Font(font_name, size)
    text_surface = font.render(text, True, color)
    text_rect = text_surface.get_rect()
    text_rect.centerx = x
    text_rect.top = y
    surf.blit(text_surface, text_rect)

def screen_init():  # 初始畫面
    global ready_blue
    global ready_red
    global test_out
    global gamestate
    screen.fill(WHITE)
    draw_text(screen, 'Welcome!', 64, BLACK, WIDTH/2, HEIGHT/8)
    draw_text(screen, ' 操作說明： ', 30, BLACK, WIDTH/4-10, HEIGHT/3+10)
    draw_text(screen, ' ↑ ↓ ← → 移動角色 ', 30, BLACK, WIDTH/2, HEIGHT/3+50)
    draw_text(screen, '~按Enter準備~', 20, BLACK, WIDTH/2, HEIGHT*7/9)
    pygame.display.update()
    waiting = True  # 等待使用者反應
    while waiting:
        if test_out == 1:
            try:s.close()
            except:pass
            try:conn.close()
            except:pass
            sys.exit(); gamestate=False; break
        for event in pygame.event.get():
            keys = pygame.key.get_pressed()
            if event.type == pygame.QUIT:
                send("out"); test_out=1
                try:s.close()
                except:pass
                try:conn.close()
                except:pass
                print("end in rule\n"); pygame.quit(); sys.exit(); break
            elif keys[pygame.K_RETURN]:
                screen.blit(block,(WIDTH/2-2*l, HEIGHT*7/9))
                draw_text(screen, '準備中', 35, (0,255,0), WIDTH/2, HEIGHT*7/9)
                pygame.display.update()
                ok='ok'
                if player == "blue": 
                    ready_blue=1
                    conn.send(ok.encode())
                elif player == "red":
                    ready_red=1
                    s.send(ok.encode())
                waiting = False
 
def screen_end():  # 結束畫面
    global gamestate
    global test_out
    screen.fill(BLACK)
    draw_text(screen, "You win!!!", 65, WHITE, WIDTH/2, HEIGHT/3)
    draw_text(screen, '按R重新開始~', 20, WHITE, WIDTH/2, HEIGHT*7/9)
    pygame.display.update()
    waiting = True  # 等待使用者反應
    while waiting:
        if test_out == 1: print("someone out, bye"); sys.exit();break
        for event in pygame.event.get():
            keys = pygame.key.get_pressed()
            if event.type == pygame.QUIT:
                send("out"); test_out=1
                try:s.close()
                except:pass
                try:conn.close()
                except:pass
                print("end in restart\n"); pygame.quit(); sys.exit(); break
            elif keys[pygame.K_r]:
                waiting = False
                screen.fill(WHITE)
                gamestate = 1
                restart()
                return False

def buttom_press(color): # 尖刺開關
    global blue_prick
    global red_prick
    if color == 'blue':
        control_Prick(blue_prick,5,1)
        renew()
    if color == 'red':
        control_Prick(red_prick,2,1)
        renew()

def cheat():
    for i in range(a+2):
        for j in range(b+2):
            if(i>a or j>b):
                pass
            elif be[i][j]==0:
                screen.blit(white, (i*l, j*l))
            elif be[i][j]==1:
                screen.blit(black, (i*l, j*l))
            elif be[i][j]==2:
                screen.blit(red, (i*l, j*l))
            elif be[i][j]==3:
                screen.blit(green, (i*l, j*l))
            elif be[i][j]==4:
                pygame.draw.circle(screen, (255,255,55), (now[0]*l+l/2,now[1]*l+l/2), R)
            elif be[i][j]==5:
                screen.blit(blue, (i*l, j*l))
    pygame.display.flip()
    time.sleep(1)
    renew()

# tcp #
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
test_out=0
test_in_blue=0
test_in_red=0
ready_blue=0
ready_red=0
conn=''; addr=''
def send(txt):
    global conn
    global s
    global player
    if txt == "out":
        out='out'
        if player == "blue":conn.send(out.encode())
        elif player == "red":s.send(out.encode())
    elif txt == "do":
        if player == "blue":
            out='blue'
            conn.send(out.encode())
        elif player == "red":
            out='red'
            s.send(out.encode())
def get_ip():
    ip = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    ip.connect(("8.8.8.8", 80))
    print("i'm at " + ip.getsockname()[0])
    return (ip.getsockname()[0])

def server():
    global test_in_blue
    global test_out
    global conn
    global addr
    global ready_red
    global gamestate
    global red_prick
    HOST = get_ip()
    PORT = 6000

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((HOST, PORT)); s.listen(1)
    print('Listening at {}'.format(s.getsockname()))
    print('等待紅方加入...') #debug: print('server start at: %s:%s' % (HOST, PORT))
    while True:
        conn, addr = s.accept()
        print('connected by ' + str(addr))
        test_in_blue=1
        while True:
            try:
                indata=conn.recv(1024)
                if indata.decode()=='ok': ready_red=1; break
                elif indata.decode() == 'out':
                    test_out=1; gamestate=False
                    conn.close(); pygame.quit(); sys.exit(); break
                else: print("error\n"); break
            except:
                print("client out\n")
                conn.close(); sys.exit(); break
                
        while True:
            if test_out == 1: break
            try:
                indata=conn.recv(1024)
                if indata.decode() == "":
                    test_out=1; gamestate=False
                    conn.close(); pygame.quit(); sys.exit(); break
                print("get_" + indata.decode())
                if indata.decode() == 'out':
                    test_out=1; gamestate=False
                    conn.close(); sys.exit(); break
                elif indata.decode() == 'red':
                    #print("i do")
                    buttom_press(indata.decode())
                    renew()
            except:
                try:s.close()
                except: pass
                try:conn.close()
                except: pass
                print("someone out, bye"); test_out=1; sys.exit(); break
def client():
    global s
    global test_out
    global test_in_red
    global ready_blue
    global gamestate
    global blue_prick
    HOST = get_ip()
    PORT = 6000
    wait_txt = 0
    while True:
        try:
            s.connect((HOST, PORT))
            print("成功加入房間\n")
            test_in_red = 1
            while True:
                try:
                    indata=s.recv(1024)
                    if indata.decode()=='ok': ready_blue=1; break
                    else: print("error"); break
                except(ConnectionResetError):
                    print("someone out\n")
                    s.close(); pygame.quit(); sys.exit(); break
            while True:
                try:
                    indata=s.recv(1024)
                    if indata.decode() == "":
                        test_out=1; gamestate=False
                        s.close(); pygame.quit(); sys.exit(); break
                    print("get_" + indata.decode())
                    if indata.decode() == 'out':
                        test_out=1; gamestate=False
                        s.close(); sys.exit(); break
                    elif indata.decode() == 'blue':
                        #print("i do")
                        buttom_press(indata.decode())
                        renew()
                except:
                    try:s.close()
                    except: pass
                    try:conn.close()
                    except: pass
                    print("someone out, bye"); test_out=1; sys.exit(); break
        except(ConnectionRefusedError):
            if (wait_txt == 0): print("等待藍方開啟..."); wait_txt = 1
            else: pass

### game ###      
while True: #身分驗證
    player=input("請輸入身分(blue/red): ")
    if(player == "blue" ):
        thread_server = threading.Thread(target=server)
        thread_server.start(); break
    elif(player == "red" ):
        thread_client = threading.Thread(target=client)
        thread_client.start(); break
    else: print("查無此身分\n")
if player == "blue": #配置地圖
    be = [
        [1,1,1,1,1,1,1,1,1,1,1,1],
        [1,4,1,1,1,0,1,1,1,5,0,1],
        [1,2,0,2,1,0,5,0,2,0,1,1],
        [1,1,1,5,1,2,1,5,1,1,1,1],
        [1,0,0,0,0,5,1,0,0,1,1,1],
        [1,1,1,2,1,1,5,1,5,2,0,1],
        [1,1,0,0,1,1,2,1,0,1,1,1],
        [1,0,5,1,1,1,0,0,2,1,1,1],
        [1,1,0,1,3,1,1,2,1,0,0,1],
        [1,1,0,1,0,5,0,0,2,5,1,1],
        [1,1,5,2,1,2,2,1,1,2,0,1],
        [1,1,1,1,1,1,1,1,1,1,1,1]]
elif player == "red":
    be = [
        [1,1,1,1,1,1,1,1,1,1,1,1],
        [1,4,0,1,1,1,3,0,2,1,1,1],
        [1,1,5,0,1,5,1,1,0,5,1,1],
        [1,1,2,2,1,0,1,0,5,2,1,1],
        [1,1,0,1,1,0,1,0,1,2,1,1],
        [1,1,0,1,2,0,5,0,1,0,1,1],
        [1,0,5,0,0,1,1,2,1,5,1,1],
        [1,2,1,0,2,0,5,1,0,2,1,1],
        [1,2,1,1,1,1,0,2,1,0,1,1],
        [1,0,5,5,2,1,1,0,1,1,0,1],
        [1,1,0,1,2,0,1,0,5,5,2,1],
        [1,1,1,1,1,1,1,1,1,1,1,1]]

## 尖刺檢測 ##
blue_prick=[]
red_prick=[]
for i in range(a+2):
    for j in range(b+2):
        if be[i][j]==2: red_prick.append([i,j])
        elif be[i][j]==5: blue_prick.append([i,j])
for i in range (len(blue_prick)):
    if i == (3 or 7 or 9 or 11 or 15):continue
    elif i%2 == 1:
        be[blue_prick[i][0]][blue_prick[i][1]] = 0
for i in range (len(red_prick)):
    if i == (2 or 8 or 12 or 6 or 20 or 14):continue
    elif i%2 == 0:
        be[red_prick[i][0]][red_prick[i][1]] = 0


while True: #連線完畢後
    time.sleep(0.05)
    if test_out == 1:
        try: s.close()
        except: pass
        try: conn.close()
        except: pass
        pygame.quit(); sys.exit(); break #print("G: someone out")
    elif gamestate == 0:  # 初始畫面
        while True:
            time.sleep(0.5)
            if test_in_blue == 1 and player == "blue":
                screen = pygame.display.set_mode((WIDTH, HEIGHT))
                pygame.display.set_caption('迷宮解謎')
                screen.fill("white")
                break
            elif test_in_red == 1 and player == "red":
                screen = pygame.display.set_mode((WIDTH, HEIGHT))
                pygame.display.set_caption('迷宮解謎')
                screen.fill("white")
                break
        screen_init()
        while True:
            try:
                if test_out == 1: print("someone out, bye"); gamestate = False; sys.exit(); break
                elif ready_blue == 1 and ready_red == 1: break
                else:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:#關閉視窗
                            send('out'); test_out=1; gamestate = False
                            print("end in ready")
                            try:s.close()
                            except:pass
                            try:conn.close()
                            except:pass
                            pygame.quit(); sys.exit(); break
                        else: pass
            except(ConnectionResetError):
                test_out=1
                try:s.close()
                except:conn.close()
                print("end in ready"); pygame.quit(); sys.exit(); break
        if test_out == 0:
            screen.fill("white")
            draw_text(screen, "進入遊戲",l, BLACK, WIDTH/2, HEIGHT/2)
            pygame.display.flip()
            for i in range(3):
                time.sleep(0.25)
                draw_text(screen, ".",l, BLACK, WIDTH/2+l*3+i*l, HEIGHT/2)
                pygame.display.flip()
                time.sleep(0.25)
            screen.fill("white")
            if player == "blue": txt_Player="blue"; color=(0,0,255)
            else: txt_Player="red"; color=(255,0,0)
            draw_text(screen, txt_Player, l, color, WIDTH-77,HEIGHT/2-(l*4))
            pygame.display.flip()
            gamestate = 1; renew()
        else: gamestate = False; break
    elif gamestate == 2: # 結束畫面
        close = screen_end()
        if close: break
    elif gamestate == 1: # 遊戲中
        if test_out == 1: print("game end\n"); pygame.quit(); sys.exit();break
        for event in pygame.event.get(): 
            if event.type == pygame.QUIT:#關閉視窗
                send("out"); test_out=1
                try:s.close()
                except:pass
                try:conn.close()
                except:pass
                print("end in game\n"); pygame.quit(); sys.exit(); break
            if event.type == pygame.MOUSEBUTTONDOWN:
                if rect1.collidepoint(event.pos):
                    pygame.draw.rect(screen, (255,0,0),rect1,2)#框
                    pygame.display.flip()
                    time.sleep(0.1)
                    buttom_press(player)
                    print(player + "_send")
                    send("do")
                    renew()
                if rect2.collidepoint(event.pos):#restart
                    pygame.draw.rect(screen, (255,0,0),rect2,2)#框
                    pygame.display.flip()
                    time.sleep(0.1)
                    restart()
                if rect3.collidepoint(event.pos):#restart
                    pygame.draw.rect(screen, (255,0,0),rect3,2)#框
                    pygame.display.flip()
                    time.sleep(0.1)
                    cheat()
            else:
                pygame.draw.rect(screen, (0,0,255),rect1,2)#框
                pygame.display.flip()

            keys = pygame.key.get_pressed()# Detect which key is pressed
            ## 移動控制 ##
            if keys[pygame.K_LEFT]: 
                if(be[now[0]-1][now[1]] != 1 and be[now[0]-1][now[1]] != 2 and be[now[0]-1][now[1]] != 5):
                    if(be[now[0]-1][now[1]]==3):
                        win(); gamestate = 2; break
                    be[now[0]][now[1]]=0
                    be[now[0]-1][now[1]]=4
                    now[0] -= 1
                    renew()
            elif keys[pygame.K_RIGHT]:
                if(be[now[0]+1][now[1]] != 1 and be[now[0]+1][now[1]] != 2 and be[now[0]+1][now[1]] != 5):
                    if(be[now[0]+1][now[1]]==3):
                        win(); gamestate = 2; break
                    be[now[0]][now[1]]=0
                    be[now[0]+1][now[1]]=4
                    now[0] += 1
                    renew()
            elif keys[pygame.K_DOWN]:
                if(be[now[0]][now[1]+1] != 1 and be[now[0]][now[1]+1] != 2 and be[now[0]][now[1]+1] != 5):
                    if(be[now[0]][now[1]+1]==3):
                        win(); gamestate = 2; break
                    be[now[0]][now[1]]=0
                    be[now[0]][now[1]+1]=4
                    now[1] += 1
                    renew()
            elif keys[pygame.K_UP]:
                if(be[now[0]][now[1]-1] != 1 and be[now[0]][now[1]-1] != 2 and be[now[0]][now[1]-1] != 5):
                    if(be[now[0]][now[1]-1]==3):
                        win(); gamestate = 2; break
                    be[now[0]][now[1]]=0
                    be[now[0]][now[1]-1]=4
                    now[1] -= 1
                    renew()
            #elif keys[pygame.K_KP_1]: send(); buttom_press('red')
            else: pass
    else: print("error, pls debug"); sys.exit(); break