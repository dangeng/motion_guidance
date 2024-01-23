////////////////////////
/// HELPER FUNCTIONS ///
////////////////////////

/*
Function to load arrows (flow) from json
Should be of format:
    [
        [dx, dy], [dx, dy], [dx, dy] ...
        [dx, dy], [dx, dy], [dx, dy] ...
        [dx, dy], [dx, dy], [dx, dy] ...
        ...
    ]
    such that data[x][y] gives offsets in x and y dirs
*/
async function loadFlow(path) {
    try {
        console.log('Loading json from: ' + path)
        const response = await fetch(path);
        const flow = await response.json();
        return flow;
    } catch (error) {
        console.error("Error loading flow:", error);
        return [];
    }
}



/////////////////////////
/// DRAWING FUNCTIONS ///
/////////////////////////

// Draws an arrow on the canvas
// From: https://codepen.io/chanthy/pen/WxQoVG
function drawArrow(canvas, fromx, fromy, tox, toy, arrowWidth, color){
    // Get canvas context
    ctx = canvas.getContext("2d");

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

// Clears the canvas
function clearCanvas(canvas) {
    canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);
}

// Draws a circle on a canvas
function drawCircle(canvas, radius, color, x, y) {
    var ctx = canvas.getContext('2d');

    ctx.beginPath();
    ctx.arc(x, y, radius, 0, 2 * Math.PI);
    ctx.strokeStyle = color;
    ctx.lineWidth = 10; // Adjust this value to change the border thickness
    ctx.stroke();
}

/*
Given:
    - an event (indicating which canvas is hovered over)
    - a triplet of canvases, representing a visualization example
    - an array of displacements (see `loadFlow`)

Draw:
    - an arrow on the flow
    - an arrow on the src image
    - circles representing corresponding points on the src and gen images
*/
function draw(event, canvases, flow) {
    // Get event canvas (either 'src' or 'flow' canvas)
    var eventCanvas = event.target;

    // Get current size of the canvas (just use 'src' canvas)
    const canvasWidth = canvases['src'].width;
    const canvasHeight = canvases['src'].height;

    // Get mouse/touch location
    // Check if it's a touch event
    if (event.touches && event.touches.length > 0) {
        x = event.touches[0].clientX;
        y = event.touches[0].clientY;
    } else {
        // It's a mouse event
        x = event.clientX;
        y = event.clientY;
    }

    // Get mouse location, relative to event canvas
    var rect = eventCanvas.getBoundingClientRect();
    const mouseX = Math.round(
                (x - rect.left) * (canvasWidth / rect.width)
            );
    const mouseY = Math.round(
                (y - rect.top) * (canvasHeight / rect.height)
            );

    // Get size of arrow
    const arrowSize = flow[mouseY][mouseX];

    // Get end point of arrow
    const arrowX = mouseX + arrowSize[0];
    const arrowY = mouseY + arrowSize[1];

    // Clear canvases
    clearCanvas(canvases['flow']);
    clearCanvas(canvases['src']);
    clearCanvas(canvases['gen']);

    // Draw circle on 'src' canvas
    drawCircle(canvases['src'], 15, 'pink', mouseX, mouseY);

    // Draw corresponding circle on 'gen' canvas
    drawCircle(canvases['gen'], 15, 'pink', arrowX, arrowY);

    // Draw arrows on 'src' and 'flow' canvases
    // only if arrow is non-zero
    if (arrowSize[0] != 0 || arrowSize[1] != 0) {
        drawArrow(
            canvases['src'], 
            mouseX, mouseY, 
            arrowX, arrowY, 
            8, 'cyan'
        );
        drawArrow(
            canvases['flow'], 
            mouseX, mouseY, 
            arrowX, arrowY, 
            8, 'black'
        );
    }
}



///////////////////////
/// SETUP FUNCTIONS ///
///////////////////////

/*
Sets up mouseleave and mousemove listeners for an example (a
triplet of canvases) and a given array of displacements (flow)
*/
function setupEventListeners(example, flow) {
    // Bind `example` and `direction` to 
    // draw function for mousemove event
    function drawPartial(event) {
        draw(event, example, flow);
    }

    // Clear all canvases on mouseleave event
    function handleMouseLeave(event) {
        clearCanvas(example['flow']);
        clearCanvas(example['src']);
        clearCanvas(example['gen']);
    }

    // Add listeners
    example['src'].addEventListener("mouseleave", handleMouseLeave);
    example['src'].addEventListener("mousemove", drawPartial);
    example['flow'].addEventListener("mouseleave", handleMouseLeave);
    example['flow'].addEventListener("mousemove", drawPartial);

    // Add listeners for touch (mobile)
    example['src'].addEventListener('touchmove', function (e) {
        // Prevent default to avoid unwanted behavior like scrolling
        e.preventDefault();
        drawPartial(e);
    });
    example['flow'].addEventListener('touchmove', function (e) {
        // Prevent default to avoid unwanted behavior like scrolling
        e.preventDefault();
        drawPartial(e);
    });
    //example['src'].addEventListener('touchend', handleMouseLeave);
}


/*
Initializes a given example (triplet of canvases):
    1. Loads the json displacement data
    2. Waits for loading to finish, then adds event listeners to canvases
*/
function initializeExample(example) {
    console.log('initializing canvas pair');
    // TODO: Remove all other data-json-path attributes
    const jsonPath = example['flow'].dataset.jsonPath;
    loadFlow(jsonPath).then(flow => {
        setupEventListeners(example, flow);
    });
}

// Initialize all examples (triplets of canvases)
function initializeExamples() {
    var examples = {};

    var canvases = document.querySelectorAll('canvas.viz');

    // Organize canvases into examples based on their common class
    canvases.forEach(function (canvas) {
        var classes = canvas.classList;
        var name = classes[1]; // name of example
        var vizType = classes[2]; // gen or src or flow

        // Initialize the index if it doesn't exist
        if (!examples[name]) {
            examples[name] = {};
        }

        // Add the canvas to the index based on both name and vizType
        examples[name][vizType] = canvas;
    });

    // Initialize all examples
    for (var name in examples) {
        var example = examples[name];
        initializeExample(example);
    };
}

// Initialize all examples on DOM loaded
document.addEventListener("DOMContentLoaded", initializeExamples);

