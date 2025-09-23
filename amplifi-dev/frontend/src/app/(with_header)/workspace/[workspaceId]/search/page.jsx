"use client";
import React, { useEffect, useState } from "react";
import DataSetFormSearch from "@/components/forms/datasetFormSearch";
import DrawerVertical from "@/components/forms/drawervertical";
import { ChevronRight } from "lucide-react";
import { Button } from "@/components/ui/button";
import Image from "next/image";
import image from "@/assets/images/empty-screens/search-empty-state.svg";
import { useParams } from "next/navigation";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { search } from "@/api/search";
import {
  identifyUserFromObject,
  hashString,
  captureEvent,
} from "@/utils/posthogUtils";
import { useUser } from "@/context_api/userContext";

const SearchPage = () => {
  const { user } = useUser();
  const params = useParams();
  const [isOpen, setIsOpen] = useState(false);
  const [selectedValue, setSelectedValue] = useState([]);
  const [searchText, setSearchText] = useState("");
  const [isSearch, setIsSearch] = useState(false);
  const [selectedOption, setSelectedOption] = useState("cosine_distance");
  const [showMetrics, setShowMetrics] = useState(false);
  const [searchData, setSearchData] = useState({});
  const [fullSearchData, setFullSearchData] = useState({}); // New state for full response
  const [showGraphOption, setshowGraphOption] = useState(true);
  const graphSearchOption = { label: "Graph Search", value: "graph_search" };

  const handleSearch = async () => {
    setIsSearch(true);

    // Track search triggered event
    identifyUserFromObject(user);
    captureEvent("search_triggered", {
      query_text: searchText,
      dataset_id: selectedValue.length > 0 ? selectedValue[0].id : "",
      model_name: selectedOption,
      user_id: hashString(user?.clientId || ""),
      description: "User clicks 'Search'",
    });

    const data = {
      id: params?.workspaceId,
      body: {
        query: searchText,
        dataset_ids: selectedValue.map((item) => item.id),
        perform_eval: true,
        perform_aggregate_search: !showMetrics,
        perform_graph_search: selectedOption === graphSearchOption.value,
        calculate_metrics: true,
        vector_search_settings: {
          search_limit: 5,
          search_index_type:
            selectedOption === graphSearchOption.value
              ? "cosine_distance"
              : selectedOption,
          probes: 10,
          ef_search: 40,
        },
      },
    };
    try {
      const response = await search(data);
      console.log("Full API response:", JSON.stringify(response, null, 2));
      console.log("Response data.data:", response?.data?.data);

      if (response.status === 200) {
        console.log("Selected option:", selectedOption);
        console.log("Show metrics:", showMetrics);
        const result = response?.data?.data || {};
        setFullSearchData(result); // Store full response
        let newSearchData;
        let resultCount = 0;

        if (showMetrics) {
          newSearchData =
            selectedOption === graphSearchOption.value
              ? result.graph_results || {}
              : result.dataset_results || []; // Use dataset_results for showMetrics

          // Calculate result count for metrics mode
          if (selectedOption === graphSearchOption.value) {
            resultCount =
              newSearchData?.graph_search_relationship_results?.length || 0;
          } else if (Array.isArray(newSearchData)) {
            resultCount = newSearchData.reduce(
              (total, dataset) =>
                total + (dataset.vector_search_results?.length || 0),
              0,
            );
          }
        } else {
          newSearchData = result.aggregate_results || {}; // For table rendering
          resultCount = newSearchData?.vector_search_results?.length || 0;
        }

        // Track search results loaded event
        captureEvent("search_results_loaded", {
          query_text: searchText,
          result_count: resultCount,
          dataset_id: selectedValue.length > 0 ? selectedValue[0].id : "",
          model_name: selectedOption,
          user_id: hashString(user?.clientId || ""),
          description: "Search results successfully returned",
        });

        console.log("Setting searchData to:", newSearchData);
        setSearchData(newSearchData);
      } else {
        console.log("Non-200 status:", response.status);
        setSearchData({});
        setFullSearchData({});
      }
    } catch (e) {
      console.error("Search request failed:", e.message);
      setSearchData({});
      setFullSearchData({});
    }
  };

  useEffect(() => {
    if (selectedValue?.length === 1) {
      const flag = selectedValue.some(
        (dataset) => dataset.knowledge_graph === true,
      );
      setshowGraphOption(flag);
      setShowMetrics(false);
    } else {
      setshowGraphOption(null);
    }
    console.log("searchData:", searchData);
    console.log("fullSearchData:", fullSearchData);
  }, [selectedValue, searchData, fullSearchData]);

  return (
    <div className="p-8">
      <div className="flex justify-between">
        <div className="font-semibold text-base flex items-center gap-2">
          Datasets :
          <div
            className="font-normal text-sm border border-gray-400 px-3 py-1 bg-white flex gap-4 items-center rounded"
            onClick={() => setIsOpen(true)}
          >
            {selectedValue.length > 0
              ? selectedValue.map((item) => item.name).join(", ")
              : "Dataset selected"}
            <ChevronRight className="font-normal text-sm w-3 h-3" />
          </div>
        </div>
        <div className="text-sm font-normal align-middle">
          <input
            type="checkbox"
            className="me-2"
            disabled={showGraphOption}
            checked={showMetrics}
            onChange={(e) => {
              setShowMetrics(e.target.checked);
              // Track show metrics toggled event
              identifyUserFromObject(user);
              captureEvent("show_metrics_toggled", {
                dataset_id: selectedValue.length > 0 ? selectedValue[0].id : "",
                user_id: hashString(user?.clientId || ""),
                state: e.target.checked ? "enabled" : "disabled",
                description: "User toggles 'Show metrics per dataset'",
              });
            }}
          />
          Show metrics per dataset
          <div className="text-xs text-gray-500 mt-1">
            Currently, images and image related datasets are not supported for
            this functionality
          </div>
        </div>
      </div>

      {isSearch ? (
        <div className="mt-5">
          <div className="text-sm font-medium">Type and query</div>
          <div className="flex justify-stretch items-center w-full">
            <div className="w-1/5">
              <select
                className="w-full p-2 border border-gray-400 rounded-s-md focus:outline-none"
                value={selectedOption}
                onChange={(e) => {
                  setSelectedOption(e.target.value);
                  // Track search model selected event
                  identifyUserFromObject(user);
                  captureEvent("search_model_selected", {
                    model_name: e.target.value,
                    dataset_id:
                      selectedValue.length > 0 ? selectedValue[0].id : "",
                    user_id: hashString(user?.clientId || ""),
                    description:
                      "User selects search model (e.g., Cosine Distance)",
                  });
                }}
              >
                <option value="cosine_distance">Cosine Distance</option>
                <option value="l2_distance">l2 Distance</option>
                <option value="ip_distance">Ip Distance</option>
                {showGraphOption && (
                  <option value={graphSearchOption.value}>
                    {graphSearchOption.label}
                  </option>
                )}
              </select>
            </div>
            <div className="w-4/5">
              <input
                type="text"
                className="w-full p-2 border border-gray-400 rounded-e-md text-sm border-s-0 focus:outline-none"
                placeholder="Promotion"
                value={searchText}
                onChange={(e) => {
                  setSearchText(e.target.value);
                  // Track search query entered event
                  if (e.target.value.trim()) {
                    identifyUserFromObject(user);
                    captureEvent("search_query_entered", {
                      query_text: e.target.value,
                      dataset_id:
                        selectedValue.length > 0 ? selectedValue[0].id : "",
                      model_name: selectedOption,
                      user_id: hashString(user?.clientId || ""),
                      description: "When user types a query",
                    });
                  }
                }}
              />
            </div>
            <Button
              className="bg-blue-10 rounded-lg font-medium text-sm px-10 ms-4"
              onClick={handleSearch}
              disabled={!selectedValue || selectedValue.length === 0}
            >
              Search
            </Button>
          </div>

          <Table className="border border-gray-300 rounded-2xl mt-8 table-fixed w-full">
            <TableHeader>
              <TableRow className="border-b-2 border-gray-300">
                <TableHead className="text-xs font-semibold bg-gray-200 text-black-10 ps-4 w-[80%]">
                  Query Results
                </TableHead>
                <TableHead className="text-xs font-semibold bg-gray-200 text-black-10 ps-4 w-[20%] text-center">
                  Search Score
                </TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {showMetrics ? (
                selectedOption === graphSearchOption.value ? (
                  searchData?.graph_search_relationship_results?.length > 0 ? (
                    searchData.graph_search_relationship_results.map(
                      (items, idx) => (
                        <TableRow
                          className="border-b-2 border-gray-300 bg-white"
                          key={idx}
                        >
                          <TableCell className="py-3 px-4 overflow-hidden whitespace-nowrap text-ellipsis text-sm w-[80%]">
                            {items.subject +
                              "-" +
                              items.predicate +
                              "-" +
                              items.object}
                          </TableCell>
                          <TableCell className="py-3 px-4 text-center text-sm w-[20%]">
                            {Number(items.score).toFixed(2)}
                          </TableCell>
                        </TableRow>
                      ),
                    )
                  ) : (
                    <TableRow>
                      <TableCell
                        colSpan={2}
                        className="py-6 text-center bg-gray-50"
                      >
                        <div className="flex flex-col items-center justify-center">
                          <p className="text-sm text-gray-500">No Data Found</p>
                          <p className="text-xs text-gray-400 mt-1">
                            Try adjusting your search or check back later.
                          </p>
                        </div>
                      </TableCell>
                    </TableRow>
                  )
                ) : Array.isArray(searchData) && searchData.length > 0 ? (
                  searchData.flatMap(
                    (dataset, datasetIdx) =>
                      dataset.vector_search_results?.map((items, idx) => (
                        <TableRow
                          className="border-b-2 border-gray-300 bg-white"
                          key={`${dataset.dataset_id}-${idx}`}
                        >
                          <TableCell className="py-3 px-4 overflow-hidden whitespace-nowrap text-ellipsis text-sm w-[80%]">
                            {items.text}
                          </TableCell>
                          <TableCell className="py-3 px-4 text-center text-sm w-[20%]">
                            {Number(items.search_score).toFixed(2)}
                          </TableCell>
                        </TableRow>
                      )) || [],
                  )
                ) : (
                  <TableRow>
                    <TableCell
                      colSpan={2}
                      className="py-6 text-center bg-gray-50"
                    >
                      <div className="flex flex-col items-center justify-center">
                        <p className="text-sm text-gray-500">No Data Found</p>
                        <p className="text-xs text-gray-400 mt-1">
                          Try adjusting your search or check back later.
                        </p>
                      </div>
                    </TableCell>
                  </TableRow>
                )
              ) : searchData?.vector_search_results?.length > 0 ? (
                searchData.vector_search_results.map((item, idx) => (
                  <TableRow
                    className="border-b-2 border-gray-300 bg-white"
                    key={idx}
                  >
                    <TableCell className="py-3 px-4 overflow-hidden whitespace-nowrap text-ellipsis text-sm w-[80%]">
                      {item.text}
                    </TableCell>
                    <TableCell className="py-3 px-4 text-center text-sm w-[20%]">
                      {Number(item.search_score).toFixed(2)}
                    </TableCell>
                  </TableRow>
                ))
              ) : (
                <TableRow>
                  <TableCell
                    colSpan={2}
                    className="py-6 text-center bg-gray-50"
                  >
                    <div className="flex flex-col items-center justify-center">
                      <p className="text-sm text-gray-500">No Data Found</p>
                      <p className="text-xs text-gray-400 mt-1">
                        Try adjusting your search or check back later.
                      </p>
                    </div>
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>

          {showMetrics ? (
            <div>
              {Array.isArray(searchData) && searchData.length > 0 ? (
                searchData.map((result, index) => (
                  <div key={index} className="my-6">
                    <div className="text-base font-semibold mt-6">
                      Metrics header
                    </div>
                    <div className="my-4 text-base font-medium">
                      Dataset:{" "}
                      {selectedValue.find(
                        (item) => item.id === result.dataset_id,
                      )?.name || result.dataset_id}
                    </div>
                    <div className="grid grid-cols-3 gap-4">
                      <div className="bg-white p-4 rounded">
                        <div className="text-2xl font-bold">
                          {result.precision_scores?.precision?.toFixed(2) ??
                            "N/A"}
                        </div>
                        <div className="text-sm font-medium">Precision</div>
                      </div>
                      <div className="bg-white p-4 rounded">
                        <div className="text-2xl font-bold">
                          {result.precision_scores?.ndcg_score?.toFixed(2) ??
                            "N/A"}
                        </div>
                        <div className="text-sm font-medium">NDCG</div>
                      </div>
                      <div className="bg-white p-4 rounded">
                        <div className="text-2xl font-bold">
                          {result.precision_scores?.latency?.toFixed(2) ??
                            "N/A"}
                        </div>
                        <div className="text-sm font-medium">Latency</div>
                      </div>
                    </div>
                  </div>
                ))
              ) : (
                <div className="my-6 text-center text-sm text-gray-500">
                  No metrics available
                </div>
              )}
            </div>
          ) : (
            selectedOption !== graphSearchOption.value && (
              <div>
                <div className="text-base font-semibold mt-6">
                  Metrics header
                </div>
                {fullSearchData?.dataset_results?.length > 0 ? (
                  fullSearchData.dataset_results.map((result, index) => (
                    <div key={index} className="my-6">
                      <div className="my-4 text-base font-medium">
                        Dataset:{" "}
                        {selectedValue.find(
                          (item) => item.id === result.dataset_id,
                        )?.name || result.dataset_id}
                      </div>
                      <div className="grid grid-cols-3 gap-4">
                        <div className="bg-white p-4 rounded">
                          <div className="text-2xl font-bold">
                            {result.precision_scores?.precision?.toFixed(2) ??
                              "N/A"}
                          </div>
                          <div className="text-sm font-medium">Precision</div>
                        </div>
                        <div className="bg-white p-4 rounded">
                          <div className="text-2xl font-bold">
                            {result.precision_scores?.ndcg_score?.toFixed(2) ??
                              "N/A"}
                          </div>
                          <div className="text-sm font-medium">NDCG</div>
                        </div>
                        <div className="bg-white p-4 rounded">
                          <div className="text-2xl font-bold">
                            {result.precision_scores?.latency?.toFixed(2) ??
                              "N/A"}
                          </div>
                          <div className="text-sm font-medium">Latency</div>
                        </div>
                      </div>
                    </div>
                  ))
                ) : searchData?.precision_scores ? (
                  <div>
                    <div className="my-4 text-base font-medium">
                      Dataset:{" "}
                      {selectedValue.find(
                        (item) =>
                          item.id ===
                          searchData?.vector_search_results?.[0]?.dataset_id,
                      )
                        ? "Aggregate Results"
                        : "N/A"}
                    </div>
                    <div className="grid grid-cols-3 gap-4">
                      <div className="bg-white p-4 rounded">
                        <div className="text-2xl font-bold">
                          {searchData.precision_scores.precision?.toFixed(2) ??
                            "N/A"}
                        </div>
                        <div className="text-sm font-medium">Precision</div>
                      </div>
                      <div className="bg-white p-4 rounded">
                        <div className="text-2xl font-bold">
                          {searchData.precision_scores.ndcg_score?.toFixed(2) ??
                            "N/A"}
                        </div>
                        <div className="text-sm font-medium">NDCG</div>
                      </div>
                      <div className="bg-white p-4 rounded">
                        <div className="text-2xl font-bold">
                          {searchData.precision_scores.latency?.toFixed(2) ??
                            "N/A"}
                        </div>
                        <div className="text-sm font-medium">Latency</div>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="my-6 text-center text-sm text-gray-500">
                    No metrics available
                  </div>
                )}
              </div>
            )
          )}
        </div>
      ) : (
        <div className="flex flex-col gap-4 w-full justify-center items-center !h-[60vh]">
          <Image src={image} alt="destination empty screen image" />
          <div className="flex gap-2 flex-col justify-center items-center">
            <div className="font-normal text-sm text-gray-800 text-center">
              Select a search model and type your query to get the search
              results
            </div>
          </div>
          <div className="flex items-center mt-4 w-2/3">
            <div className="w-2/5">
              <select
                className="w-full p-2 border border-gray-400 rounded-s-md focus:outline-none"
                value={selectedOption}
                onChange={(e) => setSelectedOption(e.target.value)}
              >
                <option value="cosine_distance">Cosine Distance</option>
                <option value="l2_distance">l2 Distance</option>
                <option value="ip_distance">Ip Distance</option>
                {showGraphOption && (
                  <option value={graphSearchOption.value}>
                    {graphSearchOption.label}
                  </option>
                )}
              </select>
            </div>
            <div className="w-3/5">
              <input
                type="text"
                className="w-full p-2 border border-gray-400 rounded-e-md text-sm border-s-0 focus:outline-none"
                placeholder="Promotion"
                value={searchText}
                onChange={(e) => setSearchText(e.target.value)}
              />
            </div>
            <Button
              className="bg-blue-10 rounded-lg font-medium text-sm px-10 ms-4"
              disabled={!selectedValue || selectedValue.length === 0}
              onClick={handleSearch}
            >
              Search
            </Button>
          </div>
        </div>
      )}

      <DrawerVertical
        isOpen={isOpen}
        onClose={() => setIsOpen(false)}
        title="Select Datasets"
      >
        <DataSetFormSearch
          setIsOpen={setIsOpen}
          selectedValue={selectedValue}
          setSelectedValue={setSelectedValue}
        />
      </DrawerVertical>
    </div>
  );
};

export default SearchPage;
