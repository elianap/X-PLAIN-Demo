import React, { useState, useEffect } from "react"
import { Redirect } from "react-router-dom"
import Container from "react-bootstrap/Container"
import Row from "react-bootstrap/Row"
import Col from "react-bootstrap/Col"
import ListGroup from "react-bootstrap/ListGroup"

function Classifiers() {
  const [classifiers, setClassifiers] = useState([])
  //const [toInstances, setToInstances] = useState(false)
  const [toAnalyses, setToAnalyses] = useState(false)

  useEffect(() => {
    async function fetchData() {
      const res = await fetch("http://127.0.0.1:5000/classifiers")
      const json = await res.json()
      setClassifiers(json)
    }

    fetchData()
  }, [])

  function postClassifier(datasetName) {
    return async () => {
      await fetch(`http://127.0.0.1:5000/classifier/${datasetName}`, {
        method: "POST"
      })
      setToAnalyses(true)
    }
  }

  if (toAnalyses) {
    return <Redirect to="/show_instances" />
  }
  return (
    <Container>
      <Row className="justify-content-md-center">
        <Col>
          <h2>Select a classifier</h2>
        </Col>
      </Row>
      <Row className="justify-content-md-center">
        <Col lg={3}>
          <ListGroup>
            {classifiers.map(classifier => (
              <ListGroup.Item
                className="text-center"
                action
                key={classifier}
                onClick={postClassifier(classifier)}
              >
                {classifier}
              </ListGroup.Item>
            ))}
          </ListGroup>
        </Col>
      </Row>
    </Container>
  )
}

export default Classifiers
