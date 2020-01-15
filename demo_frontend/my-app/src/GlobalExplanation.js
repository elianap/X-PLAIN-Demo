import React, { useState, useEffect } from "react"
import Container from "react-bootstrap/Container"
import Row from "react-bootstrap/Row"
import Col from "react-bootstrap/Col"
import Spinner from "react-bootstrap/Spinner"

import {
  GlobalExplanationPlot,
  getTrace,
  getDifferences,
  getNames
} from "./GlobalExplanationPlot"
import Nav from "react-bootstrap/Nav"

import Button from "react-bootstrap/Button"
import Octicon, { ArrowRight } from "@primer/octicons-react"
import Form from "react-bootstrap/Form"

//<Form.Label>Target class</Form.Label>

function GlobalExplanation() {
  const [globalExplanation, setGlobalExplanation] = useState(null)
  const [typeSel, setType] = useState("attribute_explanation")
  const [classValue, setClassValue] = useState("global")
  const [targetClasses, setTargetClasses] = useState([])

  const handleSelect = eventKey => {
    setType(eventKey)
  }

  useEffect(() => {
    async function fetchData() {
      const res = await fetch("http://127.0.0.1:5000/global_explanation")
      const json = await res.json()
      setGlobalExplanation(json)
      setTargetClasses(json.explainer_info.target_classes)
    }

    fetchData()
  }, [])

  function handleChangeForm(event) {
    setClassValue(event.target.value)
  }

  if (globalExplanation === null) {
    return (
      <div>
        <Container>
          <Row>
            <Col className="mt-3">
              <h2>Explanation metadata </h2>
            </Col>
          </Row>
          <Row className="justify-content-center mb-4">
            <Nav
              variant="pills justify-content-center"
              defaultActiveKey="attribute_explanation"
              onSelect={handleSelect}
            >
              <Nav.Item>
                <Nav.Link eventKey="attribute_explanation">Attribute</Nav.Link>
              </Nav.Item>
              <Nav.Item>
                <Nav.Link eventKey="attribute_value_explanation">
                  Attribute value{" "}
                </Nav.Link>
              </Nav.Item>
              <Nav.Item>
                <Nav.Link eventKey="rules_explanation">
                  Subsets of attribute values{" "}
                </Nav.Link>
              </Nav.Item>
            </Nav>
          </Row>
          <Row>
            {" "}
            <Spinner animation="border" />{" "}
          </Row>
        </Container>
      </div>
    )
  }

  const differences = getDifferences(globalExplanation[classValue][typeSel])

  const names = getNames(globalExplanation[classValue][typeSel])

  const trace = getTrace(differences, names)

  return (
    <div>
      <Container>
        <Row className="mt-3">
          <Col>
            <h2>Explanation metadata </h2>
            <p>
              Explanation metadata , dataset{" "}
              <code>{globalExplanation.explainer_info.dataset_name}</code>, of
              classifier{" "}
              <code>{globalExplanation.explainer_info.classifier_name}</code>.
            </p>
          </Col>
          <Col xs={2}>
            {" "}
            <Button
              variant="outline-dark"
              className="ml-auto p-2"
              href="/analyses"
            >
              {" "}
              <Octicon icon={ArrowRight} /> Analyses{" "}
            </Button>{" "}
          </Col>
        </Row>
        <Row className="justify-content-center mb-4">
          <Nav
            variant="pills dark justify-content-center"
            defaultActiveKey="attribute_explanation"
            onSelect={handleSelect}
          >
            <Nav.Item>
              <Nav.Link eventKey="attribute_explanation">Attribute</Nav.Link>
            </Nav.Item>
            <Nav.Item>
              <Nav.Link eventKey="attribute_value_explanation">
                Attribute value{" "}
              </Nav.Link>
            </Nav.Item>
            <Nav.Item>
              <Nav.Link eventKey="rules_explanation">
                Subsets of attribute values{" "}
              </Nav.Link>
            </Nav.Item>
          </Nav>
          <Form>
            <Form.Group controlId="exampleForm.ControlSelect1">
              <Form.Control size="sm" as="select" onChange={handleChangeForm}>
                <option> global </option>
                {targetClasses.map(target_class => (
                  <option>{target_class}</option>
                ))}
              </Form.Control>
            </Form.Group>
          </Form>
        </Row>
        {(typeSel === "rules_explanation" && trace.x.length===1) ? null: (
        <Row>
          <Col>
            <GlobalExplanationPlot
              trace={trace}
              title={
                "Dataset: " +
                globalExplanation.explainer_info.dataset_name +
                "  model=" +
                globalExplanation.explainer_info.classifier_name
              }
              xaxistitle={"Δ - target class = " + classValue}
            />
          </Col>
        </Row>
        )}
        {typeSel === "rules_explanation" ? (
          <Row>
            {" "}
            <span>
              {" "}
              {globalExplanation[classValue].rule_mapping.map(x => 
                 (globalExplanation[classValue].rule_mapping.length>1) ? 
                    (<p>{x}</p>) :
                    (<p>{x} - { globalExplanation[classValue].rules_explanation.y[0].toFixed(4)}</p>)
              )}{" "}
            </span>{" "}
          </Row>
        ) : null}
      </Container>
    </div>
  )
}

export default GlobalExplanation
