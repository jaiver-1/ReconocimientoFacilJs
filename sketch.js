let video, classifier, faceapi, inputLabel
let boxes = []
let trained = false
const imgSize = 64

function setup() {
  createCanvas(900, 520)

  initVideo()
  initFaceDetector()
  initFaceClassifier()
  drawMenu()
}

function draw() {
  background(0);
  image(video, 0, 10, 640, 520)
  drawBoxes()
}

function drawBoxes() {
  for (let i=0; i < boxes.length; i++) {
    const box = boxes[i]
    noFill()
    stroke(161, 95, 251)
    strokeWeight(4)
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
