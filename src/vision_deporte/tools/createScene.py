import xml.etree.ElementTree as ET

def build_xml(path):
    tree = ET.parse(path)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    asset = ET.SubElement(root, "asset")
    visual = ET.SubElement(root, "visual")

    ET.SubElement(visual, "rgba", {"haze" : "0.15 0.25 0.35 1"})
    
    ET.SubElement(asset, "texture", {
        "name" : "groundplane", 
        "type" : "2d", 
        "builtin" : "checker", 
        "width" : "300", 
        "height" : "300",
        "rgb1": "0.2 0.3 0.4",
        "rgb2": "0.1 0.2 0.3",
        "markrgb": "0.8 0.8 0.8"
    })
    ET.SubElement(asset, "texture", {
            "name" : "sky",
            "type": "skybox",
            "builtin": "gradient",
            "rgb1": "0.3 0.5 0.7",
            "rgb2": "0.0 0.0 0.0",
            "width": "512",
            "height": "3072"
    })
    ET.SubElement(asset, "material", {
        "name" : "groundplane", 
        "texture" : "groundplane", 
        "texuniform" : "true", 
        "texrepeat" : "5 5", 
        "reflectance" : "0.2"
    })
    
    ET.SubElement(worldbody, "light", {
        "name" : "top", 
        "pos" : "0 0 5", 
        "dir" : "0 0 -1", 
        "directional" : "true"
    })
    ET.SubElement(worldbody, "geom", {
        "name" : "floor", 
        "type" : "plane", 
        "size": "0 0 0.1", 
        "pos" : "0 0 0.0", 
        "material" : "groundplane"
    })

    return ET.tostring(root, encoding="unicode")