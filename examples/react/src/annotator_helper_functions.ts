import cv from '@techstark/opencv-js'

export const getBlobFromImage = function(inputSize: Array<number>, mean: Array<number>, std: number, swapRB: boolean, cvImg: any) {
    const matC3 = new cv.Mat(cvImg.matSize[0], cvImg.matSize[1], cv.CV_8UC3);
    cv.cvtColor(cvImg, matC3, cv.COLOR_RGBA2BGR);
    const input = cv.blobFromImage(matC3, std, new cv.Size(inputSize[0], inputSize[1]), new cv.Scalar(mean[0], mean[1], mean[2]), swapRB);
    
    matC3.delete();
    return input;
}

export const loadAnnotatorFile = async (e: any) => {
    if(!e.target.files[0]) {
      return;
    }

    return new Promise((resolve) => {
        let file = e.target.files[0];
        let path = file.name;
        let reader = new FileReader();
        reader.readAsArrayBuffer(file);
        reader.onload = function(ev) {
            if(reader.readyState === 2) {
                let buffer: any = reader.result;
                let data = new Uint8Array(buffer);
                cv.FS_createDataFile('/', path, data, true, false, false);
                resolve(path);
            }
        }
    });
}

export const generateColors = function(result: any) {
    const numClasses = result.matSize[1];
    let colors = [0, 0, 0];
    while(colors.length < numClasses * 3) {
        colors.push(Math.round((Math.random() * 255 + colors[colors.length - 3]) / 2));
    }
    return colors;
}

export const segArgmax = function(result: any, colors: Array<number>) {
    const C = result.matSize[1];
    const H = result.matSize[2];
    const W = result.matSize[3];
    const resultData = result.data32F;
    const imgSize = H*W;

    let classId = [];
    let i, j;
    for(i = 0; i < imgSize; ++i) {
        let id = 0;
        for(j = 0; j < C; ++j) {
            if(resultData[j*imgSize+i] > resultData[id*imgSize+i]) {
            id = j;
            }
        }
        classId.push(colors[id*3]);
        classId.push(colors[id*3+1]);
        classId.push(colors[id*3+2]);
        classId.push(255);
    }

    const output = cv.matFromArray(H, W, cv.CV_8UC4, classId);
    return output;
}

export const posePostProcess = function(result: any, dataset: string, threshold: number, outputWidth: number, outputHeight: number) {
    const resultData = result.data32F;
    const matSize = result.matSize;
    // const size1 = matSize[1];
    const size2 = matSize[2];
    const size3 = matSize[3];
    const mapSize = size2 * size3;

    let output = cv.Mat.zeros(outputWidth, outputHeight, cv.CV_8UC3);

    let BODY_PARTS: any = {};
    let POSE_PAIRS: any = [];

    if(dataset === 'COCO') {
        BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                    "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                    "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
                    "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 };

        POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
                    ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
                    ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
                    ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
                    ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]
    }
    else if (dataset === 'MPI') {
        BODY_PARTS = { "Head": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                    "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                    "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "Chest": 14,
                    "Background": 15 }

        POSE_PAIRS = [ ["Head", "Neck"], ["Neck", "RShoulder"], ["RShoulder", "RElbow"],
                    ["RElbow", "RWrist"], ["Neck", "LShoulder"], ["LShoulder", "LElbow"],
                    ["LElbow", "LWrist"], ["Neck", "Chest"], ["Chest", "RHip"], ["RHip", "RKnee"],
                    ["RKnee", "RAnkle"], ["Chest", "LHip"], ["LHip", "LKnee"], ["LKnee", "LAnkle"] ]
    }
    else if (dataset === 'BODY_25') {
        BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                    "LShoulder": 5, "LElbow": 6, "LWrist": 7, "MidHip": 8, "RHip": 9,
                    "RKnee": 10, "RAnkle": 11, "LHip": 12, "LKnee": 13, "LAnkle": 14,
                    "REye": 15, "LEye": 16, "REar": 17, "LEar": 18, "LBigToe": 19,
                    "LSmallToe": 20, "LHeel": 21, "RBigToe": 22, "RSmallToe": 23,
                    "RHeel": 24, "Background": 25 }

        POSE_PAIRS = [ ["Neck", "Nose"], ["Neck", "RShoulder"],
                    ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
                    ["RElbow", "RWrist"], ["LShoulder", "LElbow"],
                    ["LElbow", "LWrist"], ["Nose", "REye"],
                    ["REye", "REar"], ["Nose", "LEye"],
                    ["LEye", "LEar"], ["Neck", "MidHip"],
                    ["MidHip", "RHip"], ["RHip", "RKnee"],
                    ["RKnee", "RAnkle"], ["RAnkle", "RBigToe"],
                    ["RBigToe", "RSmallToe"], ["RAnkle", "RHeel"],
                    ["MidHip", "LHip"], ["LHip", "LKnee"],
                    ["LKnee", "LAnkle"], ["LAnkle", "LBigToe"],
                    ["LBigToe", "LSmallToe"], ["LAnkle", "LHeel"] ]
    }

    // get position of keypoints from output
    let points = [];
    let i;
    for(i = 0; i < Object.keys(BODY_PARTS).length; ++i) {
        let heatMap = resultData.slice(i * mapSize, (i+1) * mapSize);
        let maxIndex = 0;
        let maxConf = heatMap[0];
        let index: any;
        for(index in heatMap) {
            if(heatMap[index] > heatMap[maxIndex]) {
                maxIndex = index;
                maxConf = heatMap[index];
            }
        }

        if(maxConf > threshold) {
            let indexX = maxIndex % size3;
            let indexY = maxIndex / size3;

            let x = outputWidth * indexX / size3;
            let y = outputHeight * indexY / size2;

            points[i] = [Math.round(x), Math.round(y)];
        }
    }

    // draw the points and lines into the image
    for(const pair of POSE_PAIRS) {
        const partFrom = pair[0];
        const partTo = pair[1];
        const idFrom = BODY_PARTS[partFrom];
        const idTo = BODY_PARTS[partTo];
        const pointFrom = points[idFrom];
        const pointTo = points[idTo];

        if(points[idFrom] && points[idTo]) {
            cv.line(output, new cv.Point(pointFrom[0], pointFrom[1]),
                            new cv.Point(pointTo[0], pointTo[1]), new cv.Scalar(0, 255, 0), 3);
            cv.ellipse(output, new cv.Point(pointFrom[0], pointFrom[1]), new cv.Size(3, 3), 0, 0, 360,
                               new cv.Scalar(0, 0, 255), cv.FILLED);
            cv.ellipse(output, new cv.Point(pointTo[0], pointTo[1]), new cv.Size(3, 3), 0, 0, 360,
                               new cv.Scalar(0, 0, 255), cv.FILLED);
        }
    }
    return output;
}