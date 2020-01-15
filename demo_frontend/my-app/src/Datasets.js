import React, { useEffect, useState } from "react"
import { Redirect } from "react-router-dom"
import Container from "react-bootstrap/Container"
import Row from "react-bootstrap/Row"
import Col from "react-bootstrap/Col"
import ListGroup from "react-bootstrap/ListGroup"
import Form from "react-bootstrap/Form"
import Button from "react-bootstrap/Button"
import InputGroup from "react-bootstrap/InputGroup"
import FormControl from "react-bootstrap/FormControl"

function Datasets() {
  const [datasets, setDatasets] = useState([])
  const [toClassifiers, setToClassfiers] = useState(false)
  const [inputValue, setInputValue] = useState("-")

  useEffect(() => {
    async function fetchData() {
      const res = await fetch("http://127.0.0.1:5000/datasets")
      const json = await res.json()
      setDatasets(json)
    }

    fetchData()
  }, [])

  function postDataset(datasetName) {
    return async () => {
      await fetch(`http://127.0.0.1:5000/dataset/${datasetName}`, {
        method: "POST"
      })
      setToClassfiers(true)
    }
  }

  function handleChangeForm(event) {
    setInputValue(event.target.value)
  }

  if (toClassifiers) {
    return <Redirect to="/classifiers" />
  }
  return (
    <Container>
      <Row className="mt-3">
        <Col>
          <h2>Select a dataset</h2>
        </Col>
      </Row>
      <Row className="justify-content-md-center">
        <Col lg={4}>
          <ListGroup>
            {datasets.map(datasetName => (
              <ListGroup.Item
                className="text-center"
                action
                key={datasetName}
                onClick={postDataset(datasetName)}
              >
                {datasetName}
              </ListGroup.Item>
            ))}
            <ListGroup.Item>
              <Form>
                <InputGroup className="mb-3">
                  <FormControl
                    placeholder="Dataset name"
                    aria-label="Dataset name"
                    aria-describedby="basic-addon2"
                    onChange={handleChangeForm}
                  />
                  <InputGroup.Append>
                    <Button
                      variant="outline-secondary"
                      onClick={postDataset(inputValue)}
                    >
                      Select
                    </Button>
                  </InputGroup.Append>
                </InputGroup>
              </Form>
            </ListGroup.Item>
          </ListGroup>
        </Col>
      </Row>
    </Container>
  )
}

export default Datasets
