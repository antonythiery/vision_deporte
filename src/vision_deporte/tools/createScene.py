import xml.etree.ElementTree as ET

def build_xml(path):
    tree = ET.parse(path)
    root = tree.getroot()

    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.Element("compiler")
        root.insert(0, compiler)
    compiler.set("meshdir", "table/assets")
    compiler.set("texturedir", "table/assets")

    root.insert(1, ET.Element("include", {"file": "box_tests.xml"}))
    root.insert(2, ET.Element("include", {"file": "table/table.xml"}))

    return ET.tostring(root, encoding="unicode")
