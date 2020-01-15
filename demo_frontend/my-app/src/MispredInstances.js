import React, { useState, useEffect } from "react"
import Container from "react-bootstrap/Container"
import Row from "react-bootstrap/Row"
import Col from "react-bootstrap/Col"
import Spinner from "react-bootstrap/Spinner"
import { Redirect } from "react-router-dom"
import Button from "react-bootstrap/Button"
import { MyTable, makeColumns, makeClasses } from "./MyTableFunctions"

function MispredInstances() {
  function makeInstances2(response) {
    return response.mispred_instances.map(instance => {
      const row = {}
      row["id"] = instance["id"]
      response.domain.forEach((attribute, attribute_ix) => {
        row[attribute[0]] = instance[attribute[0]]
      })
      row["pred"] = instance["pred"]
      return row
    })
  }

  const [response, setResponse] = useState({})
  const [toAnalyses, setToAnalyses] = useState(false)
  const [selectedClass, setSelectedClass] = useState(null)
  const [selectedInstance, setSelectedInstance] = useState(null)

  const domain = React.useMemo(() => makeColumns(response.domain || []), [
    response.domain
  ])
  const instances = React.useMemo(
    () =>
      Object.entries(response).length === 0 ? [] : makeInstances2(response),
    [response]
  )
  const classes = React.useMemo(() => makeClasses(response.classes || []), [
    response.classes
  ])

  useEffect(() => {
    async function fetchData() {
      const res = await fetch("http://127.0.0.1:5000/mispred_instances")
      const json = await res.json()
      setResponse(json)
    }

    fetchData()
  }, [])

  function postInstance(instanceId, class_) {
    return async () => {
      await fetch(`http://127.0.0.1:5000/mispred_instance/${instanceId}`, {
        method: "POST",
        body: JSON.stringify({ class: class_ })
      })
      setToAnalyses(true)
    }
  }

  if (response.length === 0) {
    return (
      <Container>
        <Row>
          <Col className="mt-3">
            <h2>Mispredicted Instances</h2>
            <Spinner animation="border" />
          </Col>
        </Row>
      </Container>
    )
  }

  if (toAnalyses) {
    return <Redirect to="/explanation" />
  }
  return (
    <Container>
      <Row className="mt-3 d-flex align-items-center">
        <h2 className="p-2">Mispredicted Instances</h2>
        <Button
          variant="dark"
          disabled={selectedInstance === null || selectedClass === null}
          className="ml-auto p-2"
          onClick={postInstance(selectedInstance, selectedClass)}
        >
          Get Explanation
        </Button>
      </Row>
      <Row>
        <Col lg={9}>
          <h2>Select an mispredicted instance</h2>
          <MyTable
            columns={domain}
            data={instances}
            onCheck={row => e => setSelectedInstance(row.values.id)}
            isChecked={row => row.values.id === selectedInstance}
          />
        </Col>
        <Col lg={3}>
          <h2>Select a class</h2>
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
      </Row>
    </Container>
  )
}

export default MispredInstances
