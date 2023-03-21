let video, classifier, faceapi, inputLabel
let boxes = []
let trained = false
const imgSize = 64

function setup() {
  createCanvas(900, 520) // se controla el lienzo de canvas para lo cual se debe manipular funcion draw y initVideo

  initVideo()
  initFaceDetector()
  initFaceClassifier()
  drawMenu() // para menu se puede Comentar
}

function draw() {
  background(0);
  image(video, 0, 10, 640, 520) // control de tamaño video y agregar el mismo en la funcion initVideo
  drawBoxes() //para mostarr el recuadro
}

function drawBoxes() {
  for (let i=0; i < boxes.length; i++) {
    const box = boxes[i]
    noFill()
    stroke(161, 95, 251) // color del recuadro en formato rgb
    strokeWeight(4) // grosor de cuadro
    rect(box.x, box.y, box.width, box.height)

    if (box.label) {
      fill(161, 95, 251)
      rect(box.x, box.y + box.height, 100, 25)

      fill(255)
      noStroke()
      // strokeWeight(2)
      textSize(18)
      text(box.label, box.x + 10, box.y + box.height + 20)
    }
  }
}

function initVideo() {
  video = createCapture(VIDEO)
  video.size(640, 520)
  video.hide()
}

function initFaceClassifier() {
  let options = {
    inputs: [imgSize, imgSize, 4],
    task: 'imageClassification',
    debug: true,
  }
  classifier = ml5.neuralNetwork(options)
}

function initFaceDetector() {
  const detectionOptions = {
    withLandmarks: true,
    withDescriptors: false
  };

  faceapi = ml5.faceApi(video, detectionOptions, () => {
    console.log('Face API Model Loaded!')
    detectFace()
  });
}

function detectFace() {
  faceapi.detect((err, results) => {
    if (err) return console.error(err)

    boxes = []
    if (results && results.length > 0) {
      boxes = getBoxes(results)
      if (trained) {
        for (let i=0; i < boxes.length; i++) {
          const box = boxes[i]
          classifyFace(box)
        }
      }
    }
    detectFace()
  })
}

function getBoxes(detections) {
  const boxes = []
  for(let i = 0; i < detections.length; i++) {
    const alignedRect = detections[i].alignedRect

    const box = {
      x: alignedRect._box._x,
      y: alignedRect._box._y,
      width: alignedRect._box._width,
      height: alignedRect._box._height,
      label: ""
    }
    boxes.push(box)
  }

  return boxes
}

function classifyFace(box) {
  const img = get(box.x, box.y, box.width, box.height)
  img.resize(imgSize, imgSize)
  let inputImage = { image: img };
  classifier.classify(inputImage, (error, results) => {
    if (error) return console.error(error)

    // The results are in an array ordered by confidence.
    label = results[0].label
    box.label = label
  });
}

function keyPressed() {
  // if (key == " ") {
  //   addExample(inputLabel.value())
  // }
}

function addExample(label) {
  if (boxes.length === 0) {
    console.error("No face found")
  } else if (boxes.length === 1) {
    const box = boxes[0]
    img = get(box.x, box.y, box.width, box.height)
    img.resize(imgSize, imgSize)
    console.log('Adding example: ' + label)
    classifier.addData({ image: img }, { label })
  } else {
    console.error("Only one face should appear")
  }
}

function trainModel() {
  classifier.normalizeData()
  classifier.train({ epochs: 50 }, () =>  {
    console.log('training complete')
    trained = true
  })
}

function drawMenu() {
  inputLabel = createInput('')
  inputLabel.position(650, 30)

  const takePhotoBtn = createButton("Capture")
  takePhotoBtn.position(810, 30)
  takePhotoBtn.mousePressed(() => addExample(inputLabel.value()))

  const trainBtn = createButton("Train")
  trainBtn.position(650, 80)
  trainBtn.mousePressed(() => {
    trainModel()
  })

  const loadModelInput = createFileInput(file => {
    // loadData expects an array of File objects, so we have to wrap the file in an array
    classifier.loadData([file.file], () => console.log("Data Loaded"))
  })
  loadModelInput.position(650, 130)

  const saveDataBtn = createButton("Save Data")
  saveDataBtn.position(650, 180)
  saveDataBtn.mousePressed(() => classifier.saveData('model'))
}





//#region Manejo de Fotografia en otro js

/*
let Validorrecuadro = true;
function draw() {
    //background(0);
    image(video, 0, 10, 470, 400);
    recuadro();
};

function recuadro() {
    
    if (Validorrecuadro) {
        drawBoxes();
    }
    Validorrecuadro = true
};


let captureButton = document.getElementById('CapturarFoto');
captureButton.addEventListener('click', function () {

    if (boxes.length === 1) {

        Validorrecuadro = false;
        draw();

        //Obtener mi Fotografia
        let CanvaImg = document.getElementById('defaultCanvas0');
        let imagen = CanvaImg.toDataURL("image/png");
        let ImagenFormateada = imagen.replace(/^data:image\/(png|jpg);base64,/, "");

        //pintar
        let image = new Image();
        image.src = imagen;

        let canvaMarcar = document.getElementById('canvaMarca');
        let contextoMarca = canvaMarcar.getContext('2d');
        for (let i = 0; i < boxes.length; i++) {
            const Puntos = boxes[i]
            image.onload = function () {
                //drawImage(image, sx-desde donde with, sy-desde donde heig, sWidth-ancjo imagen, sHeight-altoimage, dx-dese donde empieza a dibujar, dy-dese donde empieza a dibujar, dWidth-longitud de imagen, dHeight-la altura de la imagen);
                contextoMarca.drawImage(image, Puntos.x + 50, Puntos.y - 30, 430, 800, 0, 0, 300, 400);
            }
        }
    }
    else {
        console.error("No se Encuentra Rostro");
    }

});

*/
//endregion