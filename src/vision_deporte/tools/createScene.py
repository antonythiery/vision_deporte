import xml.etree.ElementTree as ET

def build_xml(path):
    tree = ET.parse(path)
    root = tree.getroot()


    # root.insert(1, ET.Element("include", {"file": "box_tests.xml"}))
    # root.insert(2, ET.Element("include", {"file": "table/table.xml"}))
    # root.insert(3, ET.Element("include", {"file": "robot/robot.xml"}))

    return ET.tostring(root, encoding="unicode")
