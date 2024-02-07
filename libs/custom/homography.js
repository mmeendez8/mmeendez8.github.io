
// Constants
const pitchHeight = 53 + 1/3;
const pitchWidth = 120;
const lineWidth = 2; 
const numHeight = 2; // 6 feet
const numWidth = 4 / 3; // 4 feet
const numDistanceYardLine = 2/3; // 2 feet
const hashMarkLength = 2;
const hashMarkMidHeight = 6 + 6/36; // 18 feet 6 inches from the center
const pointRadius = 10;

// Elements
const imageLoader = document.getElementById('imageLoader');
const imageCanvas = document.getElementById('imageCanvas');
const pointsImageCanvas = document.getElementById('pointsImageCanvas');
const templateCanvas = document.getElementById('templateCanvas');
const pointsTemplateCanvas = document.getElementById('pointsTemplateCanvas');
const removeLastPointButton = document.getElementById('removeLastPoint');
const computeHomographyButton = document.getElementById('computeHomography');

// Contexts
const ctxImage = imageCanvas.getContext('2d');
const ctxPointsImage = pointsImageCanvas.getContext('2d');
const ctxTemplate = templateCanvas.getContext('2d');
const ctxPointsTemplate = pointsTemplateCanvas.getContext('2d');

// Variables
let scaleFactorWidth, scaleFactorHeight;
let img = new Image();
let isImageLoaded = false;
let pointsImage = [];
let pointsTemplate = [];
let lines = [];
let templateImage, sourceImage;
let dragIndex = null;

// Colors
const colors = [
    { name: 'Red', hex: '#FF0000' },
    { name: 'Green', hex: '#00FF00' },
    { name: 'Blue', hex: '#0000FF' },
    { name: 'Yellow', hex: '#FFFF00' },
    { name: 'Cyan', hex: '#00FFFF' },
    { name: 'Magenta', hex: '#FF00FF' },
    { name: 'Black', hex: '#000000' },
    { name: 'White', hex: '#FFFFFF' },
    { name: 'Gray', hex: '#808080' },
    { name: 'Light Gray', hex: '#D3D3D3' }
];

// Event listeners
imageLoader.addEventListener('change', handleImage, false);
removeLastPointButton.addEventListener('click', removeLastPoint);
computeHomographyButton.addEventListener('click', computeHomography);

// Prepare canvas for template
scaleFactorWidth = templateCanvas.offsetWidth / pitchWidth;
scaleFactorHeight = templateCanvas.offsetHeight / pitchHeight;

templateCanvas.width = scaleFactorWidth * pitchWidth + 2*lineWidth;
templateCanvas.height = scaleFactorHeight * pitchHeight + 2*lineWidth;
pointsTemplateCanvas.width = templateCanvas.width;
pointsTemplateCanvas.height = templateCanvas.height;


// Wait for opencv to load
cv.onRuntimeInitialized = function() {
    drawTemplate();

    // Load template image from canvas to opencv matrix
    let imgData = ctxTemplate.getImageData(0, 0, ctxTemplate.canvas.width, ctxTemplate.canvas.height);
    let data = new Uint8Array(imgData.data.buffer);
    templateImage = new cv.Mat(ctxTemplate.canvas.height, ctxTemplate.canvas.width, cv.CV_8UC4);
    templateImage.data.set(data);
};

function handleImage(e) {
    var reader = new FileReader();
    reader.onload = function(event) {
        img.onload = function() {
            sourceImage = cv.imread(img);
            var container = document.getElementById('canvasContainer');
            
            // Set the size of the canvas to match the image
            // mantaining image aspect ratio
            var containerWidth = container.offsetWidth;
            var aspectRatio = img.width / img.height;
            
            var newWidth = img.width;
            var newHeight = img.height;

            if (newWidth > containerWidth) {
                newWidth = containerWidth;
                newHeight = newWidth / aspectRatio;
            }

            // Set the size of the container div to match the canvas size
            container.style.width = newWidth + 'px';
            container.style.height = newHeight + 'px';

            // Set the dimensions of both canvases
            imageCanvas.width = newWidth;
            imageCanvas.height = newHeight;
            pointsImageCanvas.width = newWidth;
            pointsImageCanvas.height = newHeight;

            // Draw the image
            cv.imshow('imageCanvas', sourceImage);
        }
        img.src = event.target.result;
    }
    reader.readAsDataURL(e.target.files[0]); 

    // Add event listener to the image canvas after image is loaded
    pointsImageCanvas.addEventListener('mousedown', event => handleClick(event, pointsImageCanvas, pointsImage, ctxPointsImage));
    pointsTemplateCanvas.addEventListener('mousedown', event => handleClick(event, pointsTemplateCanvas, pointsTemplate, ctxPointsTemplate));
    pointsImageCanvas.addEventListener('mousemove', event => dragPoint(event, pointsImageCanvas, pointsImage, ctxPointsImage));
    pointsTemplateCanvas.addEventListener('mousemove', event => dragPoint(event, pointsTemplateCanvas, pointsTemplate, ctxPointsTemplate));
    pointsImageCanvas.addEventListener('mouseup', releasePoint);
    pointsTemplateCanvas.addEventListener('mouseup', releasePoint);


}


function handleClick(event, canvas, pointsArray, ctx) {
    var rect = canvas.getBoundingClientRect();
    var point = { x: event.clientX - rect.left, y: event.clientY - rect.top };

    let clickedPointIndex = clickedPoint(point, pointsArray);

    // If a point is clicked, set the dragIndex and return
    if (clickedPointIndex != null) {
        dragIndex = clickedPointIndex;
        return
    } 

    // If no point is clicked, add a new point in that position
    pointsArray.push(point);
    drawPointWithIndex(ctx, point, pointsArray.length - 1);
}

function clickedPoint(point, pointsArray) {
    // Check if a point is clicked
    let clickedPointIndex = null; 
    for (let i = 0; i < pointsArray.length; i++) {
        let p = pointsArray[i];
        let distance = Math.sqrt(Math.pow(p.x - point.x, 2) + Math.pow(p.y - point.y, 2));
        if (distance <= pointRadius) {
            clickedPointIndex = i;
            break;
        }
    }  
    return clickedPointIndex;      
}

function dragPoint(event, canvas, pointsArray, ctx) {
    if (dragIndex === null) return;

    var rect = canvas.getBoundingClientRect();
    var x = event.clientX - rect.left;
    var y = event.clientY - rect.top;

    // Update the position of the point being dragged
    pointsArray[dragIndex].x = x;
    pointsArray[dragIndex].y = y;

    // Redraw all points
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    for (var i = 0; i < pointsArray.length; i++) {
        drawPointWithIndex(ctx, pointsArray[i], i);
    }
}

function releasePoint() {
    dragIndex = null;
}

function drawPointWithIndex(ctx, point, index) {
    // Get point color by index from colors list
    var color = colors[index % colors.length];

    // Draw the point using color
    ctx.fillStyle = color.hex;
    ctx.beginPath();
    ctx.arc(point.x, point.y, pointRadius, 0, Math.PI * 2, true);
    ctx.fill();

    // Draw the index
    ctx.font = '12px Arial';
    ctx.fillText(index, point.x + 12, point.y + 12);
}

function removeLastPoint() {
    if (pointsImage.length > pointsTemplate.length && pointsImage.length > 0) {
        pointsImage.pop();
    } else if (pointsImage.length < pointsTemplate.length && pointsTemplate.length > 0) {
        pointsTemplate.pop();
    } else if (pointsImage.length === pointsTemplate.length && pointsImage.length > 0) {
        pointsImage.pop();
        pointsTemplate.pop();
    }
    redrawPoints();
}

function redrawPoints() {
    // Clear and redraw pointsImage
    ctxPointsImage.clearRect(0, 0, pointsImageCanvas.width, pointsImageCanvas.height);
    pointsImage.forEach(function(point, index) {
        drawPointWithIndex(ctxPointsImage, point, index);
    });

    // Clear and redraw pointsTemplate
    ctxPointsTemplate.clearRect(0, 0, pointsTemplateCanvas.width, pointsTemplateCanvas.height);
    pointsTemplate.forEach(function(point, index) {
        drawPointWithIndex(ctxPointsTemplate, point, index);
    });
}

function drawLine(ctx, point1, point2) {
    ctx.beginPath();
    ctx.moveTo(point1.x, point1.y);
    ctx.lineTo(point2.x, point2.y);
    ctx.stroke();
}

function addLine(point1, point2) {
    lines.push({ start: point1, end: point2 });
}

function scalePoint(point, scaleFactorWidth, scaleFactorHeight) {
    return { x: point.x * scaleFactorWidth, y: point.y * scaleFactorHeight };
}

function drawAllLines(ctx, scaleFactorWidth, scaleFactorHeight) {
    lines.forEach(line => {
        var scaledStart = scalePoint(line.start, scaleFactorWidth, scaleFactorHeight);
        var scaledEnd = scalePoint(line.end, scaleFactorWidth, scaleFactorHeight);
        drawLine(ctx, scaledStart, scaledEnd)
    });
}

function printMatrix(mat) {
    console.log("Matrix:")
    let array = Array.from(mat.data);
    for (let i = 0; i < mat.rows; i++) {
        let row = [];
        for (let j = 0; j < mat.cols; j++) {
            row.push(array[i * mat.cols + j]);
        }
        console.log(row.join(' '));
    }
}

function computeHomography() {
    // Check that both lists of points have the same size and that their size is at least 4
    if (pointsImage.length !== pointsTemplate.length || pointsImage.length < 4) {
        alert('Both lists of points must have the same size and contain at least 4 points');
        return;
    }

    // Convert points to cv.Mat format
    let imagePoints = cv.matFromArray(pointsImage.length, 1, cv.CV_32FC2, pointsImage.flatMap(point => [Math.round(point.x), Math.round(point.y)]));
    let templatePoints = cv.matFromArray(pointsTemplate.length, 1, cv.CV_32FC2, pointsTemplate.flatMap(point => [Math.round(point.x), Math.round(point.y)]));

    // Compute homography
    let homography = cv.findHomography(templatePoints, imagePoints);

    // Check if homography is none because of colinear points
    if (homography.empty())
    {
        alert("Could not found any homography for these sets of points. Be sure they are not colinear.");
        return;
    }

    // Warp the template image using the homography
    let warpedTemplate = new cv.Mat();
    cv.warpPerspective(templateImage, warpedTemplate, homography, sourceImage.size());

    // Add the warped template to the source image
    let resultWeighted = new cv.Mat();
    cv.addWeighted(sourceImage, 1, warpedTemplate, 0.5, 0, resultWeighted);
    cv.imshow('imageCanvas', resultWeighted);

    // Clean up memory
    imagePoints.delete();
    templatePoints.delete();
    warpedTemplate.delete();

    return homography;
}

function drawTextInBox(ctx, text, x, y, boxWidth, boxHeight, mirror = true) {
    let fontSize = 10; // Start with a standard font size and adjust as needed
    ctx.font = `${fontSize}px sans-serif`;

    // Increase or decrease font size to fit the box
    while (ctx.measureText(text).width < boxWidth && ctx.measureText('M').width < boxHeight) {
        fontSize++;
        ctx.font = `${fontSize}px sans-serif`;
    }

    // Once the text exceeds the box width or height, reduce the font size
    while (ctx.measureText(text).width > boxWidth || ctx.measureText('M').width > boxHeight) {
        fontSize--;
        ctx.font = `${fontSize}px sans-serif`;
    }

    // If mirror is true, flip the context horizontally 
    if (mirror) {
        ctx.save(); // Save the current state
        ctx.scale(-1, -1); // Flip the context
        x = -x - boxWidth; // Adjust the x coordinate for the flipped context
        y = -y - boxHeight;
    }

    // Draw the text in the box
    ctx.fillText(text, x, y + fontSize); // y + fontSize because y is the baseline of the text

    // If mirror is true, restore the context to its original state
    if (mirror) {
        ctx.restore();
    }
}

function drawPitchNumbers(ctx) {
    ctx.fillStyle = "red";
    const textBoxWidth = numWidth * scaleFactorWidth
    const textBoxHeight = numHeight * scaleFactorHeight
    const numbers = ["10", "20", "30", "40", "50", "40", "30", "20", "10"];
    
    for (let mirror = 0; mirror < 2; mirror++) {
        let counter = 1;

        if (mirror == 0) {
            yPos = (pitchHeight / 4  * scaleFactorHeight) - (numHeight * scaleFactorHeight)
        } else {
            yPos = (pitchHeight * (3/ 4)  * scaleFactorHeight) - (numHeight * scaleFactorHeight)
        }

        for (let number of numbers) {
            yardLine = (10 * scaleFactorWidth) + 10 * counter * scaleFactorWidth;

            if (mirror == 0) {
                number = number.split("").reverse().join("");
            }

            // iterate again over string
            for (let i = 0; i < number.length; i++) {
                // Number at the left of the yard line
                if (i == 0) {
                    xPos = yardLine - (textBoxWidth + (numDistanceYardLine * scaleFactorWidth));
                } else {
                    xPos = yardLine + (numDistanceYardLine * scaleFactorWidth)
                }
                drawTextInBox(ctx, number[i], xPos, yPos, textBoxWidth, textBoxHeight, mirror != 1 );
            }

            counter += 1;
        }
    }
}

function drawTemplate() {
    // Set size and color of template canvas
    ctxTemplate.strokeStyle = 'red';
    ctxTemplate.fillStyle = "black";
    ctxTemplate.fillRect(0, 0, templateCanvas.width, templateCanvas.height);
    // Translate to avoid cropping the sidelines
    ctxTemplate.lineWidth = lineWidth;
    ctxTemplate.translate(lineWidth, lineWidth);

    // Side lines
    addLine({ x: 0, y: 0 }, { x: 0, y: pitchHeight });
    addLine({ x: pitchWidth, y: 0 }, { x: pitchWidth, y: pitchHeight });

    // End lines
    addLine({ x: 0, y: 0 }, { x: pitchWidth, y: 0 });
    addLine({ x: 0, y: pitchHeight }, { x: pitchWidth, y: pitchHeight });

    // Yard lines
    for (let i = 10; i < pitchWidth - 5; i += 5) {
        addLine({ x: i, y: 0 }, { x: i, y: pitchHeight });
    }

    // Hash marks
    pitchCenterY = pitchHeight / 2;
    for (let i = 10; i < pitchWidth - 10; i += 1) {
        // Avoid the yard lines
        if (i % 5 != 0) {   
            // Top          
            addLine({ x: i, y: 0 }, { x: i, y: hashMarkLength });
            // Top middle
            addLine({ x: i, y: pitchCenterY - hashMarkMidHeight / 2 - hashMarkLength / 2 }, { x: i,  y: pitchCenterY - hashMarkMidHeight / 2 +  hashMarkLength / 2});
            // Bottom middle
            addLine({ x: i, y: pitchCenterY + hashMarkMidHeight / 2 - hashMarkLength / 2 }, { x: i,  y: pitchCenterY + hashMarkMidHeight / 2 +  hashMarkLength / 2});
            // Bottom
            addLine({ x: i, y: pitchHeight }, { x: i, y: pitchHeight - hashMarkLength });
        }
    }

    drawAllLines(ctxTemplate, scaleFactorWidth, scaleFactorHeight);
    drawPitchNumbers(ctxTemplate)

    templateContainer.style.height = templateCanvas.height + 'px';
    

}