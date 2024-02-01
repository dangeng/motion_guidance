import re

# Visualization component (3 canvases and images in a row)
component = '''<div class="has-text-centered" {{ id }}>
  <p class="prompt">{{ prompt }}</p>
</div>
<div class="flex">
  <div class="flexWrapper">
    <div class="outsideWrapper">
      <div class="insideWrapper">
        <img src="static/examples/{{ dirName }}/flow.png" class="canvasBGImage">
        <canvas class="viz {{ clsName }} flow" width="512" height="512"
          data-json-path="./static/examples/{{ dirName }}/flow.json"></canvas>
      </div>
    </div>
    <div class="has-text-centered">
      <p><b>Target Flow</b></p>
      <p style="font-size: 10pt;" class="hoverMe">(Hover over me)</p>
    </div>
  </div>
  <div class="flexWrapper">
    <div class="outsideWrapper">
      <div class="insideWrapper">
        <img src="static/examples/{{ dirName }}/src.png" class="canvasBGImage">
        <canvas class="viz {{ clsName }} src" width="512" height="512"></canvas>
      </div>
    </div>
    <div class="has-text-centered">
      <p><b>Source Image</b></p>
      <p style="font-size: 10pt;" class="hoverMe">(Hover over me)</p>
    </div>
  </div>
  <div class="flexWrapper">
    <div class="outsideWrapper">
      <div class="insideWrapper">
        <img src="static/examples/{{ dirName }}/gen.png" class="canvasBGImage">
        <canvas class="viz {{ clsName }} gen" width="512" height="512"></canvas>
      </div>
    </div>
    <div class="has-text-centered">
      <p><b>Motion Edited</b></p>
    </div>
  </div>
</div>'''

def makeComponent(match):
    spaces, dirName, clsName, prompt, id = match.groups()
    if id != '':
        id = f'id="{id}"'
    out = component.replace('{{ dirName }}', dirName) \
                   .replace('{{ clsName }}', clsName) \
                   .replace('{{ prompt }}', prompt) \
                   .replace('{{ id }}', id)
    return '\n'.join([f'{spaces}{line}' for line in out.splitlines()])


# Read template
with open('index.template.html', 'r+') as f:
    lines = f.readlines()
template = ''.join(lines)

# Replace tags with actual html
regex_pattern = r'( *){{(.*)\|(.*)\|(.*)\|(.*)}}'
result = re.sub(regex_pattern, makeComponent, template)

# Save to index.html
with open('index.html', 'w+') as f:
    f.write(result)