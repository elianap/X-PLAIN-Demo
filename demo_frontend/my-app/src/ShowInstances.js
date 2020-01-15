import { useTable, useSortBy, usePagination } from "react-table"
import Table from "react-bootstrap/Table"
import React, { useState, useEffect } from "react"
import Container from "react-bootstrap/Container"
import Row from "react-bootstrap/Row"
import Col from "react-bootstrap/Col"
import Spinner from "react-bootstrap/Spinner"
import Form from "react-bootstrap/Form"
import Button from "react-bootstrap/Button"
import ButtonGroup from "react-bootstrap/ButtonGroup"

import Octicon, { Graph } from "@primer/octicons-react"

function ShowInstances() {
  function MyTablev2({ columns, data }) {
    // Use the state and functions returned from useTable to build your UI
    const {
      getTableProps,
      getTableBodyProps,
      headerGroups,
      prepareRow,
      page, // Instead of using 'rows', we'll use page,
      // which has only the rows for the active page

      // The rest of these things are super handy, too ;)
      canPreviousPage,
      canNextPage,
      pageOptions,
      pageCount,
      gotoPage,
      nextPage,
      previousPage,
      setPageSize,
      state: { pageIndex, pageSize }
    } = useTable(
      {
        columns,
        data,
        initialState: { pageIndex: 0 }
      },
      useSortBy,
      usePagination
    )

    // Render the UI for your table
    return (
      <>
        <Table
          {...getTableProps()}
          hover
          style={{
            display: "block",
            overflowX: "auto",
            whiteSpace: "nowrap"
          }}
        >
          <thead>
            {headerGroups.map(headerGroup => (
              <tr {...headerGroup.getHeaderGroupProps()}>
                {headerGroup.headers.map(column => (
                  <th {...column.getHeaderProps(column.getSortByToggleProps())}>
                    {column.render("Header")}
                    <span>
                      {column.isSorted
                        ? column.isSortedDesc
                          ? " 🔽"
                          : " 🔼"
                        : ""}
                    </span>
                  </th>
                ))}
              </tr>
            ))}
          </thead>
          <tbody {...getTableBodyProps()}>
            {page.map(row => {
              prepareRow(row)
              return (
                <tr {...row.getRowProps()}>
                  {row.cells.map(cell => {
                    return (
                      <td {...cell.getCellProps()}>{cell.render("Cell")}</td>
                    )
                  })}
                </tr>
              )
            })}
          </tbody>
        </Table>
        <div>
          <div>
            <ButtonGroup className={"mr-3"}>
              <Button
                variant="dark"
                onClick={() => gotoPage(0)}
                disabled={!canPreviousPage}
              >
                {"<<"}
              </Button>{" "}
              <Button
                variant="dark"
                onClick={() => previousPage()}
                disabled={!canPreviousPage}
              >
                {"<"}
              </Button>{" "}
              <Button
                variant="dark"
                onClick={() => nextPage()}
                disabled={!canNextPage}
              >
                {">"}
              </Button>{" "}
              <Button
                variant="dark"
                onClick={() => gotoPage(pageCount - 1)}
                disabled={!canNextPage}
              >
                {">>"}
              </Button>
            </ButtonGroup>

            <span>
              Page{" "}
              <strong>
                {pageIndex + 1} of {pageOptions.length}
              </strong>{" "}
            </span>
          </div>
          <div className={"mt-3 mb-3"}>
            <Form.Row>
              <Col>
                <Form.Group>
                  <Form.Control
                    size={"sm"}
                    type="number"
                    defaultValue={pageIndex + 1}
                    onChange={e => {
                      const page = e.target.value
                        ? Number(e.target.value) - 1
                        : 0
                      gotoPage(page)
                    }}
                  />
                </Form.Group>
              </Col>
              <Col>
                <Form.Group>
                  <Form.Control
                    size={"sm"}
                    as={"select"}
                    value={pageSize}
                    onChange={e => {
                      setPageSize(Number(e.target.value))
                    }}
                  >
                    {[10, 20, 30, 40, 50].map(pageSize => (
                      <option key={pageSize} value={pageSize}>
                        Show {pageSize}
                      </option>
                    ))}
                  </Form.Control>
                </Form.Group>
              </Col>
            </Form.Row>
          </div>
        </div>
      </>
    )
  }

  function makeInstances(response) {
    return response.instances.map(instance => {
      const row = {}
      row["id"] = instance[1]
      response.domain.forEach((attribute, attribute_ix) => {
        row[attribute[0]] = attribute[1][instance[0][attribute_ix]]
      })
      return row
    })
  }

  function makeColumns(domain) {
    return [
      {
        Header: "id",
        accessor: "id"
      },
      ...domain.map(attribute => {
        const name = attribute[0]
        return {
          Header: name,
          accessor: name
        }
      })
    ]
  }

  const [response, setResponse] = useState({})

  const domain = React.useMemo(() => makeColumns(response.domain || []), [
    response.domain
  ])
  const instances = React.useMemo(
    () =>
      Object.entries(response).length === 0 ? [] : makeInstances(response),
    [response]
  )

  useEffect(() => {
    async function fetchData() {
      const res = await fetch("http://127.0.0.1:5000/show_instances")
      const json = await res.json()
      setResponse(json)
    }

    fetchData()
  }, [])

  if (response.length === 0) {
    return (
      <Container>
        <Row>
          <Col className="mt-3">
            <h2>Instances</h2>
            <Spinner animation="border" />
          </Col>
        </Row>
      </Container>
    )
  }

  return (
    <Container>
      <Row className="mt-3 d-flex align-items-center">
        <h2 className="p-2">Instances</h2>
        <Button variant="dark" className="ml-auto p-2" href="/analyses">
          {" "}
          <Octicon icon={Graph} /> Analyze{" "}
        </Button>
      </Row>
      <Row>
        <Col lg={12}>
          <h5>
            Instances of dataset <code>{response.dataset_name}</code>
          </h5>
          <MyTablev2 columns={domain} data={instances} />
        </Col>
      </Row>
    </Container>
  )
}

export default ShowInstances
