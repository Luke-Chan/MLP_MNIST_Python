from mlp import three_layer_perceptron
from http.server import HTTPServer, BaseHTTPRequestHandler
from io import BytesIO
from PIL import Image
import time
import base64
import urllib.parse

class HTTPRequestHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'hello')

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        body = self.rfile.read(content_length)
        self.send_response(200)
        self.end_headers()

        img = urllib.parse.unquote(body.decode()).replace("data:image/png;base64,", "")
        img = base64.b64decode(img)
        sample = Image.open(BytesIO(img))
        
        inputs = []
        for y in range(0, 28):
            for x in range(0, 28):
                pixel = sample.getpixel((x, y))
                if (pixel[3] == 0) :
                    inputs.append(0)
                elif ((pixel[0] + pixel[1] + pixel[2] == 0) and pixel[3] > 0) :
                    inputs.append(pixel[3] / 255)
                else :
                    inputs.append((255 - ((pixel[0] + pixel[1] + pixel[2])/3))/255)

        global perceptron
        result = perceptron.query(inputs)

        first_prob = 0
        second_prob = 0
        first_num = 0
        second_num = 0
        count = 0
        for prob in result :
            if (prob[0] > first_prob) :
                second_prob = first_prob
                second_num = first_num
                first_prob = prob[0]
                first_num = count
            count += 1

        first_prob = '%.2f' % (100 * first_prob)
        second_prob = '%.2f' % (100 * second_prob)

        response = BytesIO()
        response.write(('{"num1": %s, "num2": %s, "prob1": %s, "prob2": %s}' % (str(first_num), str(second_num), str(first_prob), str(second_prob))).encode())
        self.wfile.write(response.getvalue())

perceptron = three_layer_perceptron(784, 300, 10, 0.003)
perceptron.load("weights_input_hidden.bin", "weights_hidden_output.bin")

print("Server Start Running")

httpd = HTTPServer(('0.0.0.0', 8000), HTTPRequestHandler)
httpd.serve_forever()