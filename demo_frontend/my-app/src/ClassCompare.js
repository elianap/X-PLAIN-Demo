import React, { useState, useEffect } from "react"
import Container from "react-bootstrap/Container"
import Row from "react-bootstrap/Row"
import Col from "react-bootstrap/Col"
import Spinner from "react-bootstrap/Spinner"
import { Redirect } from "react-router-dom"
import Button from "react-bootstrap/Button"
import { MyTable, makeClasses } from "./MyTableFunctions"

function ClassCompare() {
  const [response, setResponse] = useState({})
  const [toExplanation, setToExplanation] = useState(false)

  const [selectedClass, setSelectedClass] = useState(null)
  const [selectedSecondClass, setSelectedSecondClass] = useState(null)

  const classes = React.useMemo(() => makeClasses(response.classes || []), [
    response.classes
  ])

  useEffect(() => {
    async function fetchData() {
      const res = await fetch("http://127.0.0.1:5000/class_comparison")
      const json = await res.json()
      setResponse(json)
      if (json.class_1) {
        setSelectedClass(json.class_1)
      }
    }

    fetchData()
  }, [])

  function postClasses(class1, class2) {
    return async () => {
      await fetch(`http://127.0.0.1:5000/class_comparison/${class1}`, {
        method: "POST",
        body: JSON.stringify({ class: class1, class2: class2 })
      })
      setToExplanation(true)
    }
  }

  if (response.length === 0) {
    return (
      <Container>
        <Row>
          <Col className="mt-3">
            <h2>Select target classes</h2>
            <Spinner animation="border" />
          </Col>
        </Row>
      </Container>
    )
  }

  if (toExplanation) {
    return <Redirect to="/explanation_class_comparison" />
  }
  return (
    <Container>
      <Row className="mt-3 d-flex align-items-center">
        <h2 className="p-2">Select target classes</h2>
        <Button
          variant="dark"
          disabled={selectedClass === null || selectedSecondClass === null}
          className="ml-auto p-2"
          onClick={postClasses(selectedClass, selectedSecondClass)}
        >
          Get Explanation
        </Button>
      </Row>
      <Row className="mt-3 d-flex align-items-center">
        <Col lg={2}> </Col>
        <Col lg={3}>
          <h2>1° class</h2>
          <MyTable
            columns={[
              {
                Header: "Type",
                accessor: "type"
              }
            ]}
            data={classes}
            onCheck={row => e => setSelectedClass(row.values.type)}
            isChecked={row => row.values.type === selectedClass}
          />
        </Col>
        <Col lg={3}>
          <h2>2° class</h2>
          <MyTable
            columns={[
              {
                Header: "Type",
                accessor: "type"
              }
            ]}
            data={classes}
            onCheck={row => e => setSelectedSecondClass(row.values.type)}
            isChecked={row => row.values.type === selectedSecondClass}
          />
        </Col>
        <Col lg={2}> </Col>
      </Row>
    </Container>
  )
}

export default ClassCompare
