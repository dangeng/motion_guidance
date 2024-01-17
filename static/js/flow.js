// From: https://codepen.io/chanthy/pen/WxQoVG
function drawArrow(ctx, fromx, fromy, tox, toy, arrowWidth, color){
    //variables to be used when creating the arrow
    var headlen = 10;
    var angle = Math.atan2(toy-fromy,tox-fromx);
 
    ctx.save();
    ctx.strokeStyle = color;
 
    //starting path of the arrow from the start square to the end square
    //and drawing the stroke
    ctx.beginPath();
    ctx.moveTo(fromx, fromy);
    ctx.lineTo(tox, toy);
    ctx.lineWidth = arrowWidth;
    ctx.stroke();
 
    //starting a new path from the head of the arrow to one of the sides of
    //the point
    ctx.beginPath();
    ctx.moveTo(tox, toy);
    ctx.lineTo(tox-headlen*Math.cos(angle-Math.PI/7),
               toy-headlen*Math.sin(angle-Math.PI/7));
 
    //path from the side point of the arrow, to the other side point
    ctx.lineTo(tox-headlen*Math.cos(angle+Math.PI/7),
               toy-headlen*Math.sin(angle+Math.PI/7));
 
    //path from the side point back to the tip of the arrow, and then
    //again to the opposite side point
    ctx.lineTo(tox, toy);
    ctx.lineTo(tox-headlen*Math.cos(angle-Math.PI/7),
               toy-headlen*Math.sin(angle-Math.PI/7));
 
    //draws the paths created above
    ctx.stroke();
    ctx.restore();
}

/*
Function to load arrows from json
Should be of format:
    [
        [dx, dy], [dx, dy], [dx, dy] ...
        [dx, dy], [dx, dy], [dx, dy] ...
        [dx, dy], [dx, dy], [dx, dy] ...
        ...
    ]
    such that data[x][y] gives offsets in x and y dirs
*/
async function loadArrowDirectionsFromDisk(path) {
    try {
        console.log('Loading json from: ' + path)
        const response = await fetch(path);
        const directions = await response.json();
        return directions;
    } catch (error) {
        console.error("Error loading arrow directions:", error);
        return [];
    }
}

/*
Given an array of displacements (see `loadArrowDirectionsFromDisk`)
and an event, draw an arrow on the given canvas
*/
function updateArrowPosition(event, arrowDirections) {
    // Get canvas
    var thisCanvas = event.target;

    // Get current size of the canvas
    const canvasWidth = thisCanvas.width;
    const canvasHeight = thisCanvas.height;

    // Get mouse location, relative to canvas
    var rect = thisCanvas.getBoundingClientRect();
    const mouseX = Math.round((event.clientX - rect.left) * (canvasWidth / rect.width));
    const mouseY = Math.round((event.clientY - rect.top) * (canvasHeight / rect.height));

    // Get size of arrow
    const arrowSize = arrowDirections[mouseY][mouseX];

    // If arrow is size 0, don't draw it
    if (arrowSize[0] === 0 && arrowSize[1] === 0) {
        clearCanvas(thisCanvas);
    } else {
        // Draw a new arrow
        clearCanvas(thisCanvas);
        const arrowX = mouseX + arrowSize[0];
        const arrowY = mouseY + arrowSize[1];
        drawArrow(
            thisCanvas.getContext("2d"), 
            mouseX, mouseY, 
            arrowX, arrowY, 
            4, 'black'
        );
    }
}

// Clears the canvas
function clearCanvas(canvas) {
    canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);
}

/*
Sets up mouseleave and mousemove listeners for 
a given canvas and a given array of displacements
*/
function setupEventListeners(canvas, directions) {
    // Curry the updateArrowPosition function for mousemove event
    function updateArrowPositionWithDirections(event) {
        updateArrowPosition(event, directions);
    }

    // Clear the canvas on mouseleave event
    function handleMouseLeave(event) {
        // Get canvas
        var thisCanvas = event.target;
        clearCanvas(thisCanvas);
    }

    // Add listeners
    canvas.addEventListener("mouseleave", handleMouseLeave);
    canvas.addEventListener("mousemove", updateArrowPositionWithDirections);
}


/*
Initializes a given canvas:
    1. Loads the json displacement data
    2. Waits for loading to finish, then adds event listeners to canvas
*/
function initializeCanvas(canvas) {
    console.log('initializing canvas');
    loadArrowDirectionsFromDisk(canvas.dataset.jsonPath).then(directions => {
        setupEventListeners(canvas, directions);
    });
}

// Initialize all canvases
function initializeCanvases() {
    var canvases = document.querySelectorAll('canvas.flowViz');

    canvases.forEach(function (canvas) {
        initializeCanvas(canvas);
    });
}

// Initialize all canvases on DOM loaded
//document.addEventListener("DOMContentLoaded", initializeCanvases);

