from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer

def print_msg(address, *args):
    print(address, args)

dispatcher = Dispatcher()
dispatcher.map("/*", print_msg)
dispatcher.map("/av_gesture/*", print_msg)

server = BlockingOSCUDPServer(("127.0.0.1", 9000), dispatcher)
print("Listening for OSC on 127.0.0.1:9000")
server.serve_forever()