import React, { useState, useEffect } from "react"
import Container from "react-bootstrap/Container"
import Row from "react-bootstrap/Row"
import Col from "react-bootstrap/Col"
import Spinner from "react-bootstrap/Spinner"
import Button from "react-bootstrap/Button"

import Octicon, { Graph } from "@primer/octicons-react"

import Rules from "./Rules"
import {
  ExplanationPlot,
  getTrace,
  getDifferences,
  getNames
} from "./ExplanationPlot"

function Explanation() {
  const [explanation, setExplanation] = useState(null)

  useEffect(() => {
    async function fetchData() {
      const res = await fetch("http://127.0.0.1:5000/explanation")
      const json = await res.json()
      setExplanation(json)
    }

    fetchData()
  }, [])

  if (explanation === null) {
    return (
      <Container>
        <Row>
          <Col className="mt-3">
            <h2>Explanation</h2>
            <Spinner animation="border" />
          </Col>
        </Row>
      </Container>
    )
  }

  const differences = getDifferences(explanation)

  const names = getNames(explanation)

  const trace = getTrace(differences, names)

  return (
    <Container>
      <Row className="mt-3 mb-3 d-flex align-items-center">
        <Col>
          <h2>Explanation</h2>
          <p>
            The instance <code>{explanation.instance_id}</code> of dataset{" "}
            <code> {explanation.explainer_info.dataset_name} </code> belongs to
            the class <b>{explanation.target_class}</b> with probability{" "}
            <code>{explanation.prob.toFixed(3)}</code>. True class:{" "}
            <code>{explanation.true_class}</code>
          </p>
        </Col>
        <Col xs={2}>
          {" "}
          <Button
            variant="outline-dark"
            className="ml-auto p-2"
            href="/analyses_new"
          >
            {" "}
            <Octicon icon={Graph} /> New analyses{" "}
          </Button>{" "}
        </Col>
      </Row>
      <Row>
        <Col>
          <Rules explanation={explanation} />
        </Col>
        <Col xs={7}>
          <ExplanationPlot
            trace={trace}
            title={
              "Dataset: " +
              explanation.explainer_info.dataset_name +
              "  model=" +
              explanation.explainer_info.classifier_name +
              "<br>p(y=" +
              explanation.target_class +
              "|" +
              explanation.explainer_info.meta +
              ")=" +
              explanation.prob.toFixed(3) +
              "  true class=" +
              explanation.true_class
            }
            xaxistitle={"Δ - target class = " + explanation.target_class}
          />
        </Col>
      </Row>
    </Container>
  )
}

export default Explanation
